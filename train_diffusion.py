import torch, argparse, numpy as np
from torch.distributed.optim import ZeroRedundancyOptimizer
from nerf.network import NeRFNetwork
from nerf.renderer import NeRFRenderer
from nerf.provider import get_loaders
from nerf.utils import seed_everything, PSNRMeter
from diffusion.gaussian_diffusion import GaussianDiffusion, get_beta_schedule
from diffusion.unet import UNetModel
from diffusion.utils import Trainer


class DiffusionModel(torch.nn.Module):
    def __init__(self, opt, criterion, fp16=False, device=None):
        super().__init__()

        self.opt = opt
        self.criterion = criterion
        self.device = device

        self.betas = get_beta_schedule('linear', beta_start=0.0001, beta_end=self.opt.beta_end, num_diffusion_timesteps=1000)
        self.diffusion_process = GaussianDiffusion(betas=self.betas)

        attention_resolutions = (int(self.opt.coarse_volume_resolution / 4), int(self.opt.coarse_volume_resolution / 8))
        channel_mult = [int(it) for it in self.opt.channel_mult.split(',')]
        assert len(channel_mult) == 4

        self.diffusion_network = UNetModel(
            in_channels=self.opt.coarse_volume_channel,
            model_channels=self.opt.model_channels,
            out_channels=self.opt.coarse_volume_channel,
            num_res_blocks=self.opt.num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=0.0,
            channel_mult=channel_mult,
            dims=3,
            use_checkpoint=True,
            use_fp16=fp16,
            num_head_channels=64,
            use_scale_shift_norm=True,
            resblock_updown=True,
            encoder_channels=512,
        )
        self.diffusion_network.to(self.device)

    def forward(self, x, t, cond):
        if self.opt.low_freq_noise > 0:
            alpha = self.opt.low_freq_noise
            noise = np.sqrt(1 - alpha) * torch.randn_like(x) + np.sqrt(alpha) * torch.randn(x.shape[0], x.shape[1], 1, 1, 1, device=x.device, dtype=x.dtype)
        else:
            noise = torch.randn_like(x)

        x_t = self.diffusion_process.q_sample(x, t, noise=noise)
        x_pred = self.diffusion_network(x_t, t, cond)
        loss = self.criterion(x, x_pred)

        return loss, x_pred

    def get_params(self, lr):
        params = [
            {'params': list(self.diffusion_network.parameters()), 'lr': lr},
        ]
        return params


def load_encoder(opt, device):
    volume_network = NeRFNetwork(opt=opt, device=device)
    volume_renderer = NeRFRenderer(opt=opt, network=volume_network, device=device)
    volume_renderer_checkpoint = torch.load(opt.encoder_ckpt, map_location='cpu')
    volume_renderer_state_dict = {}
    for k, v in volume_renderer_checkpoint['model'].items():
        volume_renderer_state_dict[k.replace('module.', '')] = v
    volume_renderer.load_state_dict(volume_renderer_state_dict)
    volume_renderer.eval()
    volume_encoder = volume_renderer.network.encoder
    return volume_encoder, volume_renderer


def fn(i, opt):
    world_size, global_rank, local_rank = opt.gpus * opt.nodes, i + opt.node * opt.gpus, i

    if world_size > 1:
        torch.distributed.init_process_group(backend='nccl', init_method=f'tcp://{opt.master}:{opt.port}', world_size=world_size, rank=global_rank)

    if local_rank == 0:
        print(opt)

    print(f'initiate node{opt.node}, rank{global_rank}, gpu{local_rank}')
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(local_rank)
    seed_everything(opt.seed + global_rank)

    train_ids = open(opt.path, 'r').read().strip().splitlines()
    val_ids = train_ids[:opt.validate_objects]
    test_ids = open(opt.test_list, 'r').read().splitlines()[:8]

    vol_batch_size, opt.batch_size = opt.batch_size, 1
    train_loader, val_loader, test_loader = get_loaders(opt, train_ids, val_ids, test_ids, batch_size=vol_batch_size)

    volume_encoder, volume_renderer = load_encoder(opt, device)

    criterion = torch.nn.MSELoss(reduction='none')

    diffusion_model = DiffusionModel(opt, criterion, fp16=opt.fp16, device=device)
    diffusion_model.to(device)

    optimizer = ZeroRedundancyOptimizer(
        diffusion_model.get_params(opt.lr),
        optimizer_class=torch.optim.Adam,
        betas=(0.9, 0.99),
        eps=1e-6,
        weight_decay=2e-3,
        parameters_as_bucket_view=False,
        overlap_with_ddp=False,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1)

    trainer = Trainer(name='train',
                      opt=opt,
                      device=device,
                      metrics=[PSNRMeter()],
                      optimizer=optimizer,
                      scheduler=scheduler,
                      criterion=criterion,
                      model=diffusion_model,
                      encoder=volume_encoder,
                      renderer=volume_renderer,
                      clip_model="ViT-B/32",
                      ema_decay=opt.ema_decay,
                      eval_interval=opt.eval_interval,
                      workspace=opt.save_dir,
                      checkpoint_path=opt.ckpt,
                      local_rank=global_rank,
                      world_size=world_size,
                      )
    trainer.train(train_loader, val_loader, test_loader, opt.epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('save_dir', type=str)

    # data
    parser.add_argument('--data_root', type=str, default='path/to/dataset')
    parser.add_argument('--test_list', type=str, default='path/to/test_object_list')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--validate_objects', type=int, default=8)
    parser.add_argument('--downscale', type=int, default=1)

    # training
    parser.add_argument('--gpus', type=int, default=8)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--node', type=int, default=0)
    parser.add_argument('--master', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=12345)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--ckpt', type=str, default='scratch')
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--ema_decay', type=float, default=0.99)
    parser.add_argument('--ema_freq', type=int, default=10)
    parser.add_argument('--depth_loss', type=float, default=0)
    parser.add_argument('--lpips_loss', type=float, default=0)

    # encoder
    parser.add_argument('--image_channel', type=int, default=3)
    parser.add_argument('--extractor_channel', type=int, default=32)
    parser.add_argument('--coarse_volume_resolution', type=int, default=32)
    parser.add_argument('--coarse_volume_channel', type=int, default=4)
    parser.add_argument('--fine_volume_channel', type=int, default=32)
    parser.add_argument('--gaussian_lambda', type=float, default=1e4)
    parser.add_argument('--n_source', type=int, default=32)
    parser.add_argument('--mlp_layer', type=int, default=5)
    parser.add_argument('--mlp_dim', type=int, default=256)
    parser.add_argument('--costreg_ch_mult', type=str, default='2,4,8')
    parser.add_argument('--encoder_clamp_range', type=float, default=100)
    parser.add_argument('--encoder_ckpt', type=str, default='encoder.pth')

    # diffusion
    parser.add_argument('--beta_end', type=float, default=0.03)
    parser.add_argument('--model_channels', type=int, default=128)
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--channel_mult', type=str, default='1,2,3,5')
    parser.add_argument('--timestep_range', type=str, default='0,1000')
    parser.add_argument('--timestep_to_eval', type=str, default='-1')
    parser.add_argument('--low_freq_noise', type=float, default=0.5)
    parser.add_argument('--encoder_mean', type=float, default=-4.15856266)
    parser.add_argument('--encoder_std', type=float, default=4.82153749)
    parser.add_argument('--diffusion_clamp_range', type=float, default=3)

    # render
    parser.add_argument('--num_rays', type=int, default=24576)
    parser.add_argument('--num_steps', type=int, default=256)
    parser.add_argument('--bound', type=float, default=1)

    opt = parser.parse_args()
    torch.multiprocessing.spawn(fn, args=(opt,), nprocs=opt.gpus)
