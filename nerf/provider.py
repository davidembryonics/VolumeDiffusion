import os, cv2, json, torch, random, numpy as np
from PIL import Image


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=1.0, offset=[0, 0, 0]):
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose


@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, patch=False):
    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = torch.meshgrid(torch.linspace(0, W - 1, W, device=device), torch.linspace(0, H - 1, H, device=device), indexing='ij') # 
    i = i.t().reshape([1, H * W]).expand([B, H * W]) + 0.5
    j = j.t().reshape([1, H * W]).expand([B, H * W]) + 0.5

    results = {}

    if N > 0:
        if patch:
            assert H == W
            grid_size = int(H / 4)
            offset = [
                (0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2),
                (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1),
            ]
            patch_offset = [random.choice(offset) for _ in range(B)]
            patch_mask = torch.zeros(B, H, W, device=device)
            for k in range(B):
                patch_mask[k, patch_offset[k][0] * grid_size : (patch_offset[k][0] + 2) * grid_size, patch_offset[k][1] * grid_size : (patch_offset[k][1] + 2) * grid_size] = 1
            patch_mask = patch_mask > 0
            patch_mask = patch_mask.reshape(B, -1)

            inds = torch.arange(0, H * W, device=device).unsqueeze(0).repeat(B, 1)
            patch_inds = inds[patch_mask].reshape(B, -1)

            N = N - grid_size ** 2 * 4
            if N > 0:
                rand_inds = inds[~patch_mask].reshape(B, -1)
                rand_inds = torch.gather(rand_inds, -1, torch.randint(0, rand_inds.shape[1], size=[B, N], device=device))
                inds = torch.cat([patch_inds, rand_inds], dim=-1)
            else:
                inds = patch_inds
                
            i = torch.gather(i, -1, inds)
            j = torch.gather(j, -1, inds)
            results['inds'] = inds
        else:
            N = min(N, H * W)
            inds = torch.randint(0, H * W, size=[N], device=device) # may duplicate
            inds = inds.expand([B, N])
            i = torch.gather(i, -1, inds)
            j = torch.gather(j, -1, inds)
            results['inds'] = inds
    else:
        inds = torch.arange(H * W, device=device).expand([B, H * W])

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2)
    rays_o = poses[..., :3, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d)
    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    return results


class NeRFDataset:
    def __init__(self, opt, root_path, all_ids, device, split='train', scale=1.0):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.split = split
        self.scale = scale
        self.downscale = self.opt.downscale
        self.root_path = root_path
        self.all_ids = all_ids

        self.training = self.split in ['train', 'all', 'trainval']
        self.num_rays = self.opt.num_rays if self.training else -1
        self.n_source = opt.n_source

        self.batch_size = self.opt.batch_size
        self.num_frames = 40

        self.image_size = 256

        with open(os.path.join(self.root_path, self.all_ids[0], 'meta', '000000.json'), 'r') as f:
            meta = json.load(f)['cameras'][0] 
            self.focal_x = meta['focal_length'] / meta['sensor_width'] * self.image_size
            self.focal_y = meta['focal_length'] / meta['sensor_width'] * self.image_size
        self.intrinsics = [self.focal_x, self.focal_y, self.image_size / 2, self.image_size / 2]
    
    def __len__(self):
        if self.training:
            return len(self.all_ids)
        elif self.split == 'test':
            return len(self.all_ids) * 10

    def load_views(self, id, idx, num_tgt):
        poses, images, depths = [], [], []

        for i in idx:
            image_size = self.image_size if len(poses) >= num_tgt else int(self.image_size / self.downscale)

            with open(os.path.join(self.root_path, id, 'meta', f'{i:06d}.json'), 'r') as f:
                meta = json.load(f)['cameras'][0]
            pose = np.array(meta['transformation'], dtype=np.float32)
            pose = nerf_matrix_to_ngp(pose, scale=2*self.scale)
            poses.append(pose)

            image_path = os.path.join(self.root_path, id, 'image', '{:06d}.png'.format(i))
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image.shape[-1] == 3: 
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            if image.shape[0] != image_size or image.shape[1] != image_size:
                image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
            image = image.astype(np.float32) / 255
            images.append(image)

            depth = np.array(Image.open(os.path.join(self.root_path, id, 'depth', '{:06d}.png'.format(i))))
            depth[depth > 254] = 0
            depth = np.array(Image.fromarray(depth).resize((image_size, image_size), Image.Resampling.BILINEAR)).astype(np.float32) / 100 * 2
            depths.append(depth)

        tgt_poses, tgt_images, tgt_depths = np.stack(poses[:num_tgt], axis=0), np.stack(images[:num_tgt], axis=0), np.stack(depths[:num_tgt], axis=0)
        tgt_poses, tgt_images, tgt_depths = torch.from_numpy(tgt_poses), torch.from_numpy(tgt_images), torch.from_numpy(tgt_depths)
        tgt_poses, tgt_images, tgt_depths = tgt_poses.float(), tgt_images.float(), tgt_depths.float()

        ref_poses, ref_images, ref_depths = np.stack(poses[num_tgt:], axis=0), np.stack(images[num_tgt:], axis=0), np.stack(depths[num_tgt:], axis=0)
        ref_poses, ref_images, ref_depths = torch.from_numpy(ref_poses), torch.from_numpy(ref_images), torch.from_numpy(ref_depths)
        ref_poses, ref_images, ref_depths = ref_poses.float(), ref_images.float(), ref_depths.float()

        return self.intrinsics, tgt_poses, tgt_images, tgt_depths, ref_poses, ref_images, ref_depths

    def __getitem__(self, index):

        if self.split == 'test':

            obj_id = index // 10
            tgt_idx = index % 10

            if 1 + self.n_source <= self.num_frames:
                idx = torch.randperm(self.num_frames - 1)[:self.n_source] + 1
                idx = torch.cat((torch.tensor([0]), idx), dim=0)
                idx = (idx + tgt_idx) % self.num_frames
                assert tgt_idx not in idx[1:]
            else:
                tgt_idx = torch.tensor([tgt_idx])
                ref_idx = torch.randperm(self.num_frames)[:self.n_source]
                idx = torch.cat((tgt_idx, ref_idx), dim=0)

            intrinsics, tgt_poses, tgt_images, tgt_depths, ref_poses, ref_images, ref_depths = self.load_views(self.all_ids[obj_id], idx, 1)
        
            rays = get_rays(tgt_poses, [it / self.downscale for it in intrinsics], int(self.image_size / self.downscale), int(self.image_size / self.downscale))

            results = {
                'H': self.image_size,
                'W': self.image_size,
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],
                'obj_id': obj_id,
                'ref_img': ref_images,
                'ref_pose': ref_poses,
                'ref_depth': ref_depths,
                'intrinsic': intrinsics,
                'raw_images': tgt_images.clone(),
                'raw_depths': tgt_depths.clone(),
                'images': tgt_images,
                'depths': tgt_depths,
                'id': self.all_ids[obj_id],
                'idn': obj_id,
                'idx': idx,
                'index': index
            }

            results['caption'] = open(os.path.join(self.root_path, self.all_ids[obj_id], 'caption.txt'), 'r').read().strip()

            return results
        
        elif self.split == 'train':
        
            obj_id = index

            if self.batch_size + self.n_source <= self.num_frames:
                idx = torch.randperm(self.num_frames)[:self.batch_size+self.n_source]
            else:
                tgt_idx = torch.randperm(self.num_frames)[:self.batch_size]
                ref_idx = torch.randperm(self.num_frames)[:self.n_source]
                idx = torch.cat((tgt_idx, ref_idx), dim=0)
            
            intrinsics, tgt_poses, tgt_images, tgt_depths, ref_poses, ref_images, ref_depths = self.load_views(self.all_ids[obj_id], idx, self.batch_size)

            rays = get_rays(tgt_poses, [it / self.downscale for it in intrinsics],
                            int(self.image_size / self.downscale), int(self.image_size / self.downscale),
                            self.num_rays, patch = self.opt.lpips_loss > 0)

            results = {
                'H': self.image_size,
                'W': self.image_size,
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],
                'raw_images': tgt_images.clone(),
                'raw_depths': tgt_depths.clone(),
                'obj_id': obj_id,
                'ref_img': ref_images,
                'ref_pose': ref_poses,
                'ref_depth': ref_depths,
                'intrinsic': intrinsics,
                'id': self.all_ids[obj_id],
                'idn': obj_id,
                'idx': idx,
                'index': index
            }

            C = tgt_images.shape[-1]
            results['images'] = torch.gather(tgt_images.view(self.batch_size, -1, C), 1, torch.stack(C * [rays['inds']], -1))
            results['depths'] = torch.gather(tgt_depths.view(self.batch_size, -1, 1), 1, torch.stack(1 * [rays['inds']], -1))

            results['caption'] = open(os.path.join(self.root_path, self.all_ids[obj_id], 'caption.txt'), 'r').read().strip()

            return results


def collate(x):
    if len(x) == 1:
        return x[0]
    else:
        ret = list(x)
        return ret


def get_loaders(opt, train_ids, val_ids, test_ids, batch_size=1):
    device = torch.device('cpu')

    train_dataset = NeRFDataset(opt, root_path=opt.data_root, all_ids=train_ids, device=device, split='train')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if opt.gpus > 1 else None
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=8, collate_fn=collate)

    val_dataset = NeRFDataset(opt, root_path=opt.data_root, all_ids=val_ids, device=device, split='test')
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False) if opt.gpus > 1 else None
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, sampler=val_sampler, num_workers=4, collate_fn=collate)

    test_dataset = NeRFDataset(opt, root_path=opt.data_root, all_ids=test_ids, device=device, split='test')
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False) if opt.gpus > 1 else None
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, sampler=test_sampler, num_workers=4, collate_fn=collate)

    return train_loader, val_loader, test_loader
