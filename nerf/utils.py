import os, tqdm, random, tensorboardX, time, torch, lpips, numpy as np
from PIL import Image
from rich.console import Console
from diffusion.ema_utils import ExponentialMovingAverage


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    #torch.backends.cudnn.deterministic = True


class PSNRMeter:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths)

        psnr = -10 * np.log10(np.mean((preds - truths) ** 2))
        
        self.V += psnr
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar('PSNR/' + prefix, self.measure(), global_step)

    def report(self):
        return f'PSNR = {self.measure():.6f}'


class Trainer(object):
    def __init__(self, 
                 name, # name of this experiment
                 opt, # extra conf
                 model, # network 
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer for mlp
                 scheduler=None, # scheduler for mlp
                 ema_decay=None, # if use EMA, set the decay
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 eval_interval=1, # eval once every $ epoch
                 workspace='workspace', # workspace to save logs & ckpts
                 checkpoint_path="scratch", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 ):
        
        self.name = name
        self.opt = opt
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.eval_interval = eval_interval
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        self.device = device if device is not None else torch.device(f'cuda:{local_rank%8}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()

        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(self.workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")
            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            os.makedirs(self.ckpt_path, exist_ok=True)

        if self.opt.lpips_loss > 0:
            self.lpips = lpips.LPIPS(net='vgg')
            self.lpips.to(self.device)

        if isinstance(criterion, torch.nn.Module):
            criterion.to(self.device)
        self.criterion = criterion
 
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.opt.fp16)

        self.model = model
        self.model.to(self.device)
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, find_unused_parameters=False)

        if ema_decay > 0:
            self.ema = ExponentialMovingAverage(self.model, decay=ema_decay, device=torch.device('cpu'))
        else:
            self.ema = None

        if self.workspace is not None:
            if checkpoint_path == "scratch":
                self.log("[INFO] Training from scratch ...")
            else:
                if self.local_rank == 0:
                    self.log(f"[INFO] Loading {checkpoint_path} ...")
                self.load_checkpoint(checkpoint_path)

        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
            
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.opt.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] Model Parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

    def __del__(self):
        if self.log_ptr: 
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            self.console.print(*args, **kwargs)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)
                self.log_ptr.flush()

    def train(self, train_loader, valid_loader, test_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name), flush_secs=30)

        self.evaluate_one_epoch(valid_loader, name='train')
        self.evaluate_one_epoch(test_loader, name='test')

        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch
            self.train_one_epoch(train_loader)

            if self.local_rank == 0:
                self.save_checkpoint()

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader, name='train')
                self.evaluate_one_epoch(test_loader, name='test')
 
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()
    
    def prepare_data(self, data):
        ret = {}
        for k, v in data.items():
            if type(v) is torch.Tensor:
                ret[k] = v.to(self.device)
            else:
                ret[k] = v
        return ret
    
    def step(self, data, eval=False):
        data = self.prepare_data(data)

        if eval:
            forward_fn = self.model.module.staged_forward if self.world_size > 1 else self.model.staged_forward
        else:
            forward_fn = self.model.forward
        outputs = forward_fn(
            data['rays_o'], data['rays_d'],
            ref_img=data['ref_img'], ref_pose=data['ref_pose'], ref_depth=data['ref_depth'], intrinsic=data['intrinsic'],
            bg_color=0
        )

        B, H, W, _ = data['raw_images'].shape
        if eval:
            pred_rgb = outputs['image'].reshape(B, H, W, 3).contiguous()
            pred_depth = outputs['depth'].reshape(B, H, W).contiguous()
            gt_rgb = data['images'][..., :3].reshape(B, H, W, 3).contiguous()
            gt_depth = data['depths'].reshape(B, H, W).contiguous()
        else:
            pred_rgb = outputs['image'].reshape(-1).contiguous()
            pred_depth = outputs['depth'].reshape(-1).contiguous()
            gt_rgb = data['images'][..., :3].reshape(-1).contiguous()
            gt_depth = data['depths'].reshape(-1).contiguous()

        loss_rgb = self.criterion(pred_rgb, gt_rgb).mean().reshape(-1).contiguous()
        loss_depth = self.criterion(pred_depth, gt_depth).mean().reshape(-1).contiguous()
        loss = loss_rgb + self.opt.depth_loss * loss_depth
        if self.opt.lpips_loss > 0:
            if eval:
                _gt_rgb, _pred_rgb = gt_rgb.permute(0, 3, 1, 2).contiguous(), pred_rgb.permute(0, 3, 1, 2).contiguous()
            else:
                _H, _W = 128, 128
                _gt_rgb = data['images'][:, :_H*_W, :3].reshape(B, _H, _W, 3).permute(0, 3, 1, 2).contiguous()
                _pred_rgb = pred_rgb.reshape(B, -1, 3)[:, :_H*_W, :3].reshape(B, _H, _W, 3).permute(0, 3, 1, 2).contiguous()
            loss_lpips = self.lpips.forward(_pred_rgb, _gt_rgb, normalize=True)
            loss_lpips = loss_lpips.mean().reshape(-1).contiguous()
            loss = loss + loss_lpips * self.opt.lpips_loss
        loss = loss.mean().reshape(-1).contiguous()

        ret = {
            'loss': loss,
            'loss_rgb': loss_rgb,
            'loss_depth': loss_depth,
            'pred_rgb': pred_rgb,
            'pred_depth': pred_depth,
            'gt_rgb': gt_rgb,
            'gt_depth': gt_depth,
        }

        if self.opt.lpips_loss > 0:
            ret['loss_lpips'] = loss_lpips

        return loss, ret

    def train_one_epoch(self, loader):
        self.log(f"==> Training epoch {self.epoch}, lr_mlp={self.optimizer.param_groups[0]['lr']:.6f}, lr_encoder={self.optimizer.param_groups[1]['lr']:.6f}")

        total_loss, total_loss_rgb, total_loss_depth, total_loss_lpips = 0, 0, 0, 0

        self.model.train()

        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader), bar_format='{desc} {percentage:2.1f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        data_iter = iter(loader)
        start_time = time.time()
        for _ in range(len(loader)):
            data = next(data_iter)
            
            self.local_step += 1
            self.global_step += 1
            
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                loss, loss_detail = self.step(data)
        
            self.scaler.scale(loss).backward()

            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val
            loss_val_rgb = loss_detail['loss_rgb'].item()
            total_loss_rgb += loss_val_rgb
            loss_val_depth = loss_detail['loss_depth'].item()
            total_loss_depth += loss_val_depth
            if self.opt.lpips_loss > 0:
                loss_val_lpips = loss_detail['loss_lpips'].item()
                total_loss_lpips += loss_val_lpips

            if self.ema is not None and self.global_step % self.opt.ema_freq == 0:
                self.ema.update()

            if self.local_rank == 0:
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/loss_rgb", loss_val_rgb, self.global_step)
                    self.writer.add_scalar("train/loss_depth", loss_val_depth, self.global_step)
                    if self.opt.lpips_loss > 0:
                        self.writer.add_scalar("train/loss_lpips", loss_val_lpips, self.global_step)

                if self.opt.lpips_loss > 0:
                    pbar.set_description(f"loss={loss_val:.6f}({total_loss/self.local_step:.6f}), rgb={loss_val_rgb:.6f}({total_loss_rgb/self.local_step:.6f}), depth={loss_val_depth:.6f}({total_loss_depth/self.local_step:.6f}), lpips={loss_val_lpips:.6f}({total_loss_lpips/self.local_step:.6f}), lr_mlp={self.optimizer.param_groups[0]['lr']:.6f}, lr_encoder={self.optimizer.param_groups[1]['lr']:.6f} ")
                else:
                    pbar.set_description(f"loss={loss_val:.6f}({total_loss/self.local_step:.6f}), rgb={loss_val_rgb:.6f}({total_loss_rgb/self.local_step:.6f}), depth={loss_val_depth:.6f}({total_loss_depth/self.local_step:.6f}), lr_mlp={self.optimizer.param_groups[0]['lr']:.6f}, lr_encoder={self.optimizer.param_groups[1]['lr']:.6f} ")
                pbar.update()
        
        if self.local_rank == 0 and self.use_tensorboardX:
            self.writer.flush()

        average_loss = total_loss / self.local_step

        epoch_time = time.time() - start_time
        self.log(f"\n==> Finished epoch {self.epoch} | loss {average_loss} | time {epoch_time}")

    def evaluate_one_epoch(self, loader, name=None):
        if name is None:
            name = self.name

        self.log(f"++> Evaluate name {name} epoch {self.epoch} step {self.global_step}")

        out_folder = f'ep{self.epoch:04d}_step{self.global_step:08d}/{name}'

        total_loss, total_loss_rgb, total_loss_depth, total_loss_lpips = 0, 0, 0, 0

        for metric in self.metrics:
            metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc} {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0

            for data in loader:
                _, ret = self.step(data, eval=name)

                reduced_ret = {}
                for k, v in ret.items():
                    v_list = [torch.zeros_like(v, device=self.device) for _ in range(self.world_size)]
                    torch.distributed.all_gather(v_list, v)
                    reduced_ret[k] = torch.cat(v_list, dim=0)

                loss_val = reduced_ret['loss'].mean().item()
                total_loss += loss_val
                loss_val_rgb = reduced_ret['loss_rgb'].mean().item()
                total_loss_rgb += loss_val_rgb
                loss_val_depth = reduced_ret['loss_depth'].mean().item()
                total_loss_depth += loss_val_depth
                if 'loss_lpips' in reduced_ret:
                    loss_val_lpips = reduced_ret['loss_lpips'].mean().item()
                    total_loss_lpips += loss_val_lpips

                for metric in self.metrics:
                    metric.update(reduced_ret['pred_rgb'], reduced_ret['gt_rgb'])
                
                keys_to_save = ['pred_rgb', 'gt_rgb', 'pred_depth', 'gt_depth']
                save_suffix = ['rgb.png', 'rgb_gt.png', 'depth.png', 'depth_gt.png']

                if self.local_rank == 0:
                    os.makedirs(os.path.join(self.workspace, 'validation', out_folder), exist_ok=True)
                    for k, n in zip(keys_to_save, save_suffix):
                        vs = reduced_ret[k]
                        for i in range(vs.shape[0]):
                            file_name = f'{self.local_step*self.world_size+i+1:04d}_{n}'
                            save_path = os.path.join(self.workspace, 'validation', out_folder, file_name)
                            v = vs[i].detach().cpu()
                            if 'depth' in k:
                                v = v / 5.1
                                if 'gt' in k:
                                    v[v > 1] = 0
                            v = (v.clip(0, 1).numpy() * 255).astype(np.uint8)
                            img = Image.fromarray(v)
                            img.save(save_path)

                self.local_step += 1
                if self.local_rank == 0:
                    if 'loss_lpips' in reduced_ret:
                        pbar.set_description(f"loss={loss_val:.6f}({total_loss/self.local_step:.6f}), rgb={loss_val_rgb:.6f}({total_loss_rgb/self.local_step:.6f}), depth={loss_val_depth:.6f}({total_loss_depth/self.local_step:.6f}), lpips={loss_val_lpips:.6f}({total_loss_lpips/self.local_step:.6f}) ")
                    else:
                        pbar.set_description(f"loss={loss_val:.6f}({total_loss/self.local_step:.6f}), rgb={loss_val_rgb:.6f}({total_loss_rgb/self.local_step:.6f}), depth={loss_val_depth:.6f}({total_loss_depth/self.local_step:.6f}) ")
                    pbar.update()

        if self.local_rank == 0:
            pbar.close()

            if len(self.metrics) > 0:
                for i, metric in enumerate(self.metrics):
                    self.log(metric.report(), style="blue")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.global_step, prefix=name)
                    metric.clear()

            if self.use_tensorboardX:
                self.writer.flush()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluated name {name} epoch {self.epoch} step {self.global_step}")

    def save_checkpoint(self, name=None, full=True):
        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}_step{self.global_step:08d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model': self.model.state_dict(),
        }

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['scheduler'] = self.scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()

        file_path = f"{self.ckpt_path}/{name}.pth"
        torch.save(state, file_path)

    def load_checkpoint(self, checkpoint=None):

        checkpoint_dict = torch.load(checkpoint, map_location='cpu')
        
        model_state_dict = checkpoint_dict['model']

        missing_keys, unexpected_keys = self.model.load_state_dict(model_state_dict, strict=False)
        self.log("[INFO] Loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] Missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] Unexpected keys: {unexpected_keys}")   

        if self.ema is not None and 'ema' in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict['ema'])
        
        optimizer_and_scheduler = {
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
        }

        if self.opt.fp16:
            optimizer_and_scheduler['scaler'] = self.scaler

        for k, v in optimizer_and_scheduler.items():
            if v and k in checkpoint_dict:
                try:
                    v.load_state_dict(checkpoint_dict[k])
                    self.log(f"[INFO] Loaded {k}.")
                except:
                    self.log(f"[WARN] Failed to load {k}.")
