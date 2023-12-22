import torch, argparse
from nerf.network import NeRFNetwork
from nerf.renderer import NeRFRenderer
from nerf.provider import get_loaders
from nerf.utils import seed_everything, PSNRMeter, Trainer


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

    train_loader, val_loader, test_loader = get_loaders(opt, train_ids, val_ids, test_ids)

    network = NeRFNetwork(opt=opt, device=device)
    model = NeRFRenderer(opt=opt, network=network, device=device)
    criterion = torch.nn.MSELoss(reduction='none')

    optimizer = torch.optim.Adam(model.network.get_params(opt.lr0, opt.lr1), betas=(0.9, 0.99), eps=1e-6)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1)

    trainer = Trainer(name='train',
                      opt=opt,
                      device=device,
                      metrics=[PSNRMeter()],
                      optimizer=optimizer,
                      scheduler=scheduler,
                      criterion=criterion,
                      model=model,
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
    parser.add_argument('--batch_size', type=int, default=1)
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
    parser.add_argument('--lr0', type=float, default=1e-3)
    parser.add_argument('--lr1', type=float, default=1e-4)
    parser.add_argument('--ckpt', type=str, default='scratch')
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--ema_decay', type=float, default=0)
    parser.add_argument('--ema_freq', type=int, default=10)
    parser.add_argument('--depth_loss', type=float, default=0)
    parser.add_argument('--lpips_loss', type=float, default=0.01)
    
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

    # render
    parser.add_argument('--num_rays', type=int, default=24576)
    parser.add_argument('--num_steps', type=int, default=256)
    parser.add_argument('--bound', type=float, default=1)

    opt = parser.parse_args()
    torch.multiprocessing.spawn(fn, args=(opt,), nprocs=opt.gpus)
