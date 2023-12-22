import torch


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # bins: [B, T], old_z_vals
    # weights: [B, T - 1], bin weights.
    # return: [B, n_samples], new_z_vals

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).to(weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (B, n_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


@torch.cuda.amp.autocast(enabled=False)
def near_far_from_bound(rays_o, rays_d, bound, type='cube', min_near=0.05):
    # rays: [B, N, 3], [B, N, 3]
    # bound: int, radius for ball or half-edge-length for cube
    # return near [B, N, 1], far [B, N, 1]

    radius = rays_o.norm(dim=-1, keepdim=True)

    if type == 'sphere':
        near = radius - bound # [B, N, 1]
        far = radius + bound

    elif type == 'cube':
        tmin = (-bound - rays_o) / (rays_d + 1e-15) # [B, N, 3]
        tmax = (bound - rays_o) / (rays_d + 1e-15)
        near = torch.where(tmin < tmax, tmin, tmax).max(dim=-1, keepdim=True)[0]
        far = torch.where(tmin > tmax, tmin, tmax).min(dim=-1, keepdim=True)[0]
        # if far < near, means no intersection, set both near and far to inf (1e9 here)
        mask = far < near
        near[mask] = 1e9
        far[mask] = 1e9
        # restrict near to a minimal value
        near = torch.clamp(near, min=min_near)

    return near, far


class NeRFRenderer(torch.nn.Module):
    def __init__(self, opt, network, device,):
        super().__init__()

        self.network = network
        self.device = device
        self.opt = opt

        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        # NOTE: aabb (can be rectangular) is only used to generate points, we still rely on bound (always cubic) to calculate density grid and hashing.
        aabb_train = torch.tensor([-opt.bound, -opt.bound, -opt.bound, opt.bound, opt.bound, opt.bound], dtype=torch.float32, device=self.device)
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)

    def forward(self, rays_o, rays_d,
                ref_img=None, ref_pose=None, ref_depth=None, intrinsic=None,
                bg_color=0, volume=None):

        B = rays_o.shape[0]
        prefix = rays_o.shape[:-1]
        rays_o = rays_o.reshape(B, -1, 3).contiguous()
        rays_d = rays_d.reshape(B, -1, 3).contiguous()

        N = rays_o.shape[1] # N = B * N, in fact
        device = rays_o.device

        results = {}

        aabb = self.aabb_train if self.training else self.aabb_infer

        nears, fars = near_far_from_bound(rays_o, rays_d, self.opt.bound)

        z_vals = torch.linspace(0.0, 1.0, self.opt.num_steps, device=device).reshape(1, 1, -1) # [B, 1, T]
        z_vals = z_vals.repeat(1, N, 1) # [B, N, T]
        z_vals = nears + (fars - nears) * z_vals # [B, N, T], in [nears, fars]
        sample_dist = (fars - nears) / (self.opt.num_steps - 1) # [B, N, T]

        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1) # [B, N, 1, 3] * [B, N, T, 1] -> [B, N, T, 3]
        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:]) # a manual clip.

        dirs = rays_d.unsqueeze(-2).repeat(1, 1, self.opt.num_steps, 1) # [B, N, T, 3]

        outputs, volume = self.network(xyzs.reshape(B, -1, 3), dirs.reshape(B, -1, 3), ref_img, ref_pose, ref_depth, intrinsic, volume=volume)
        for k, v in outputs.items():
            outputs[k] = v.view(B, N, self.opt.num_steps, -1)

        deltas = z_vals[..., 1:] - z_vals[..., :-1] # [B, N, T-1]
        deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
        alphas = 1 - torch.exp(-deltas * outputs['sigma'].squeeze(-1)) # [B, N, T]
        alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [B, N, T+1]
        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [B, N, T]

        rgbs = outputs['color']
        rgbs = rgbs.reshape(B, N, -1, 3) # [B, N, T, 3]

        weights_sum = weights.sum(dim=-1) # [B, N]
        
        depth = torch.sum(weights * z_vals, dim=-1) # [B, N]

        image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2) # [B, N, 3], in [0, 1]

        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color

        image = image.view(*prefix, 3)
        depth = depth.view(*prefix)
        weights_sum = weights_sum.reshape(*prefix)
        
        results['image'] = image
        results['depth'] = depth
        results['weights'] = weights
        results['weights_sum'] = weights_sum
        
        return results

    def staged_forward(self, rays_o, rays_d, ref_img, ref_pose, ref_depth, intrinsic, bg_color=0, volume=None, max_ray_batch=4096):

        if volume is None:
            with torch.no_grad():
                volume = self.network.encoder.project_volume(ref_img, ref_pose, ref_depth, intrinsic)

        B, N = rays_o.shape[:2]
        depth = torch.empty((B, N), device=self.device)
        image = torch.empty((B, N, 3), device=self.device)
        weights_sum = torch.empty((B, N), device=self.device)

        for b in range(B):
            head = 0
            while head < N:
                tail = min(head + max_ray_batch, N)
                with torch.no_grad():
                    results_ = self.forward(rays_o[b:b+1, head:tail], rays_d[b:b+1, head:tail], bg_color=bg_color, volume=volume)
                depth[b:b+1, head:tail] = results_['depth']
                weights_sum[b:b+1, head:tail] = results_['weights_sum']
                image[b:b+1, head:tail] = results_['image']
                head += max_ray_batch
                
        results = {}
        results['depth'] = depth
        results['image'] = image
        results['weights_sum'] = weights_sum

        return results

