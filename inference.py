import torch, argparse, os, glob, shutil, tqdm, clip, numpy as np
from PIL import Image
from nerf.network import NeRFNetwork
from nerf.renderer import NeRFRenderer
from nerf.provider import get_rays
from diffusion.gaussian_diffusion import GaussianDiffusion, get_beta_schedule
from diffusion.unet import UNetModel
from diffusion.dpmsolver import NoiseScheduleVP, model_wrapper, DPM_Solver


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
        x = self.diffusion_network(x, t, cond)
        return x
    
    def load_ckpt(self):
        ckpt = torch.load(self.opt.diffusion_ckpt, map_location='cpu')
        if not self.opt.dont_use_ema and 'ema' in ckpt:
            state_dict = {}
            for i, n in enumerate(ckpt['ema']['parameter_names']):
                state_dict[n.replace('module.', '')] = ckpt['ema']['shadow_params'][i]
        else:
            state_dict = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}
        self.load_state_dict(state_dict)


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


def get_clip_embedding(clip_model, text):
    x = clip_model.token_embedding(text).type(clip_model.dtype)
    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)
    return x


def circle_poses(device, radius=1.5, theta=60, phi=0):
    def safe_normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    theta = theta / 180 * np.pi * torch.ones([], device=device)
    phi = phi / 180 * np.pi * torch.ones([], device=device)
    centers = torch.stack([
        torch.sin(theta) * torch.sin(phi),
        torch.cos(theta),
        torch.sin(theta) * torch.cos(phi),
    ], dim=-1).to(device).unsqueeze(0)
    centers = safe_normalize(centers) * radius

    forward_vector = - safe_normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers
    return poses


def main(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print('[ 1/10] load encoder')

    volume_encoder, volume_renderer = load_encoder(opt, device)

    print('[ 2/10] load diffusion model')

    diffusion_model = DiffusionModel(opt, criterion=None, fp16=opt.fp16, device=device)
    diffusion_model.to(device)
    diffusion_model.load_ckpt()
    diffusion_model.eval()

    print('[ 3/10] prepare text embedding')

    clip_model, _ = clip.load('ViT-B/32', device=device)
    clip_model.eval()

    text_token = clip.tokenize([opt.prompt]).to(device)
    text_embedding = get_clip_embedding(clip_model, text_token).permute(0, 2, 1).contiguous()
    text_embedding = text_embedding.to(device).to(torch.float32)

    print('[ 4/10] prepare solver')

    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.from_numpy(diffusion_model.betas).to(device))

    model_fn = model_wrapper(
        diffusion_model,
        noise_schedule,
        model_type='x_start',
        model_kwargs={'cond': text_embedding},
    )

    dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type='dpmsolver++')

    ch, res = opt.coarse_volume_channel, opt.coarse_volume_resolution
    if opt.low_freq_noise > 0:
        alpha = opt.low_freq_noise
        noise = np.sqrt(1 - alpha) * torch.randn(1, ch, res, res, res, device=device) + np.sqrt(alpha) * torch.randn(1, ch, 1, 1, 1, device=device, dtype=torch.float32)
    else:
        noise = torch.randn(1, ch, res, res, res, device=device)

    print('[ 5/10] generate volume')

    volume = dpm_solver.sample(
        x=noise,
        steps=111,
        t_start=1.0,
        t_end=1/1000,
        order=3,
        skip_type='time_uniform',
        method='multistep',
    )

    volume = volume.clamp(-opt.diffusion_clamp_range, opt.diffusion_clamp_range)
    volume = volume * opt.encoder_std + opt.encoder_mean
    volume = volume.clamp(-opt.encoder_clamp_range, opt.encoder_clamp_range)
    volume = volume_encoder.super_resolution(volume)

    print('[ 6/10] save volume')

    out_path = os.path.join('./gen', opt.prompt_refine.replace(' ', '_'))
    os.makedirs(os.path.join(out_path, 'image'), exist_ok=True)

    open(os.path.join(out_path, 'prompt.txt'), 'w').write(f'prompt for diffusion: {opt.prompt}\nprompt for refine: {opt.prompt_refine}\n')
    torch.save(volume, os.path.join(out_path, 'volume.pth'))

    print('[ 7/10] render images')

    res = opt.render_resolution
    focal = 35 / 32 * res * 0.5
    intrinsics = [focal, focal, res / 2, res / 2]

    for i in tqdm.trange(opt.num_rendering):
        pose = circle_poses(device, radius=2.0, theta=70, phi=int(i / opt.num_rendering * 360))
        rays = get_rays(pose, intrinsics, res, res, -1)

        outputs = volume_renderer.staged_forward(
            rays['rays_o'], rays['rays_d'],
            ref_img=None, ref_pose=None, ref_depth=None, intrinsic=None,
            bg_color=0, volume=volume,
        )

        pred_rgb = outputs['image'].reshape(res, res, 3).contiguous()
        pred_depth = outputs['depth'].reshape(res, res).contiguous()

        pred_rgb = (pred_rgb.clip(0, 1).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(pred_rgb).save(os.path.join(out_path, 'image', f'{i}_rgb.png'))

        pred_depth = ((pred_depth / 5.1).clip(0, 1).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(pred_depth).save(os.path.join(out_path, 'image', f'{i}_depth.png'))
    
    return volume, volume_renderer


def convert(opt, volume, encoder):
    ckpt = {'epoch': 0, 'global_step': 0}
    ckpt['state_dict'] = {
        'geometry.encoding.encoding.volume': volume.transpose(2, 3).transpose(3, 4).flip(3),
        'renderer.estimator.occs': torch.ones(32768, dtype=torch.float32),
        'renderer.estimator.binaries': torch.ones((1, 32, 32, 32), dtype=torch.bool),
    }

    for i in [0, 2, 4, 6, 8]:
        v = encoder.network.sigma_net.net[i].weight
        ckpt['state_dict'][f'geometry.density_network.layers.{i}.weight'] = v[:1] if i == 8 else v
        ckpt['state_dict'][f'geometry.feature_network.layers.{i}.weight'] = v[1:] if i == 8 else v
        v = encoder.network.sigma_net.net[i].bias
        ckpt['state_dict'][f'geometry.density_network.layers.{i}.bias'] = v[:1] if i == 8 else v
        ckpt['state_dict'][f'geometry.feature_network.layers.{i}.bias'] = v[1:] if i == 8 else v

    torch.save(ckpt, os.path.join('./gen', opt.prompt_refine.replace(' ', '_'), 'converted_for_refine.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--prompt_refine', type=str, default=None)
    parser.add_argument('--encoder_ckpt', type=str, default='encoder.pth')
    parser.add_argument('--diffusion_ckpt', type=str, default='diffusion.pth')
    parser.add_argument('--num_rendering', type=int, default=8)
    parser.add_argument('--render_resolution', type=int, default=512)
    parser.add_argument('--dont_use_ema', action='store_true')
    parser.add_argument('--fp16', action='store_true')

    # encoder
    parser.add_argument('--image_channel', type=int, default=3)
    parser.add_argument('--extractor_channel', type=int, default=32)
    parser.add_argument('--coarse_volume_resolution', type=int, default=32)
    parser.add_argument('--coarse_volume_channel', type=int, default=4)
    parser.add_argument('--fine_volume_channel', type=int, default=32)
    parser.add_argument('--gaussian_lambda', type=float, default=1e4)
    parser.add_argument('--mlp_layer', type=int, default=5)
    parser.add_argument('--mlp_dim', type=int, default=256)
    parser.add_argument('--costreg_ch_mult', type=str, default='2,4,8')
    parser.add_argument('--encoder_clamp_range', type=float, default=100)

    # diffusion
    parser.add_argument('--beta_end', type=float, default=0.03)
    parser.add_argument('--model_channels', type=int, default=128)
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--channel_mult', type=str, default='1,2,3,5')
    parser.add_argument('--low_freq_noise', type=float, default=0.5)
    parser.add_argument('--encoder_mean', type=float, default=-4.15856266)
    parser.add_argument('--encoder_std', type=float, default=4.82153749)
    parser.add_argument('--diffusion_clamp_range', type=float, default=3)

    # render
    parser.add_argument('--num_rays', type=int, default=24576)
    parser.add_argument('--num_steps', type=int, default=512)
    parser.add_argument('--upsample_steps', type=int, default=512)
    parser.add_argument('--bound', type=float, default=1)

    opt = parser.parse_args()

    opt.prompt_refine = opt.prompt if opt.prompt_refine is None else opt.prompt_refine

    save_name = opt.prompt_refine.replace(' ', '_')

    volume, encoder = main(opt)

    print('[ 8/10] convert checkpoint for refine')

    convert(opt, volume, encoder)

    print('[ 9/10] refine with threestudio')

    os.system(f'cd ./threestudio; CUDA_VISIBLE_DEVICES=0 python launch.py --config ../refine/refine.yaml --train --gpu 0 system.prompt_processor.prompt="{opt.prompt_refine}" system.weights=../gen/{save_name}/converted_for_refine.pth')

    print('[10/10] collect results')

    output = sorted(list(glob.glob(f'./threestudio/outputs/refine/{save_name}*')))[-1]

    shutil.copytree(os.path.join(output, 'ckpts'), os.path.join('./gen', save_name, 'threestudio-ckpt'))
    shutil.copytree(os.path.join(output, 'save'), os.path.join('./gen', save_name, 'threestudio-save'))
    shutil.copy(os.path.join('./gen', save_name, 'threestudio-save', 'it1000-test.mp4'), os.path.join('./gen', save_name, 'video.mp4'))

    print(f'Done! Results are now in ./gen/{save_name}')
    print(f'Take a look at ./gen/{save_name}/video.mp4 for your generation!')
