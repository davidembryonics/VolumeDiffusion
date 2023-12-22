import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .v2v import V2VNet, V2VNetSR


class NormAct(nn.Module):
    def __init__(self, channel):
        super(NormAct, self).__init__()
        self.bn = nn.BatchNorm2d(channel)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.bn(x)
        x = self.act(x)
        return x


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=NormAct):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_act(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class SmallNetwork(nn.Module):
    def __init__(self, in_channel=3, out_channel=32):
        super(SmallNetwork, self).__init__()
        self.conv = nn.Sequential(
                        ConvBnReLU(in_channel, int(out_channel // 2), 5, 2, 2),
                        ConvBnReLU(int(out_channel // 2), out_channel, 5, 2, 2),
                    )
        self.toplayer = nn.Conv2d(out_channel, out_channel, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.toplayer(x)
        return x


class ExtractorNet(nn.Module):
    def __init__(self, device, in_channel=3, out_channel=32, checkpoint=False):
        super(ExtractorNet, self).__init__()
        self.checkpoint = checkpoint
        self.in_channel = in_channel
        self.net = SmallNetwork(in_channel, out_channel)
        self.net.to(device)
    
    def forward(self, input):
        input = input.permute(0, 3, 1, 2).contiguous()[:, :self.in_channel, :, :]
        out = checkpoint(self.net, input, use_reentrant=False) if self.checkpoint else self.net(input)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out


class CostRegNet(nn.Module):
    def __init__(self, device, model='unet', in_channel=32, out_channel=32, ch_mult=(1,2,4), checkpoint=True):
        super(CostRegNet, self).__init__()
        self.model = model
        self.checkpoint = checkpoint
        if self.model == 'v2v':
            self.net = V2VNet(in_channel, out_channel, ch_mult=ch_mult)
        elif self.model == 'v2vsr':
            self.net = V2VNetSR(in_channel, out_channel)
        self.net.to(device)
    
    def forward(self, input):
        while len(input.shape) < 5:
            input = input.unsqueeze(0)
        if self.model == 'v2vsr':
            dummy = torch.zeros([1,], device=input.device, requires_grad=True)
            out = checkpoint(self.net, input, dummy, use_reentrant=False) if self.checkpoint else self.net(input, dummy)
        else:
            out = checkpoint(self.net, input, use_reentrant=False) if self.checkpoint else self.net(input)
        return out.squeeze()


class Encoder(nn.Module):
    def __init__(self, device=None, opt=None):
        super(Encoder, self).__init__()
        self.device = device
        self.opt = opt
        self.input_dim = self.opt.image_channel
        self.extractor_channel = self.opt.extractor_channel
        self.unproject_volume_channel = self.extractor_channel * 2 + 2
        self.coarse_volume_channel = self.opt.coarse_volume_channel
        self.fine_volume_channel = self.opt.fine_volume_channel
        self.bbox = self.opt.bound
        self.clamp_range = self.opt.encoder_clamp_range

        self.volume = None
        self.extractor = ExtractorNet(device=self.device, in_channel=self.input_dim, out_channel=self.extractor_channel)
        self.costreg = CostRegNet(device=self.device, model='v2v', in_channel=self.unproject_volume_channel, out_channel=self.coarse_volume_channel, ch_mult=[int(it) for it in self.opt.costreg_ch_mult.split(',')])
        self.sr_net = CostRegNet(device=self.device, model='v2vsr', in_channel=self.coarse_volume_channel, out_channel=self.fine_volume_channel, ch_mult=(1,1))

    def generate_volume_features(self, p, volume):
        xyz_new = p.clip(-1.0 + 1e-6, 1.0 - 1e-6)

        xyz_new = xyz_new.unsqueeze(-2).unsqueeze(-2)
        while len(volume.shape) < 5:
            volume = volume.unsqueeze(0)
        volume = volume.repeat(xyz_new.shape[0], 1, 1, 1, 1)
        cxyz = F.grid_sample(volume, xyz_new, align_corners=False)

        cxyz = cxyz.squeeze(-1).squeeze(-1).transpose(1, 2)
        return cxyz

    def project_volume(self, ref_img, ref_pose, ref_depth, intrinsic, raw_volume=False):
        res = self.opt.coarse_volume_resolution
        gaussian = int(self.opt.gaussian_lambda / 64 * res)

        intrinsic = torch.tensor([[intrinsic[0] / 256 * 64 / res, 0., 0., 0.],
                                  [0., intrinsic[1] / 256 * 64 / res, 0., 0.],
                                  [0., 0., 1., 0.]], device=self.device, dtype=torch.float32)
        x = torch.linspace(-self.bbox, self.bbox, res, device=self.device)
        x, y, z = torch.meshgrid(x, x, x, indexing='ij')
        xyz = torch.stack((x, y, z, torch.ones_like(x)), dim=-1).permute(3, 0, 1, 2).reshape(4, -1)

        volume, variance = 0, 0
        in_mask = torch.zeros((1, 1, res, res, res), device=self.device)
        max_in_mask = torch.zeros((1, 1, res, res, res), device=self.device)

        feat = self.extractor(ref_img)
        feat = feat.permute(0, 3, 1, 2)

        for i in range(len(ref_img)):
            __feat = feat[i:i+1]

            uv = (intrinsic @ torch.linalg.inv(ref_pose[i]) @ xyz).permute(1, 0)
            depth = uv[:, 2]
            uv = uv / uv[:, 2:] * 1
            uv = uv[:, :2].unsqueeze(0).unsqueeze(2)

            _feat = F.grid_sample(__feat, uv, align_corners=False, padding_mode='zeros').squeeze()

            _depth = F.grid_sample(ref_depth[i].unsqueeze(0).unsqueeze(0), uv, align_corners=False, padding_mode='zeros').squeeze()
            _in_mask = torch.exp(-1 * gaussian * (depth - _depth) ** 2) * 1e4

            _feat = _feat.reshape(1, self.extractor_channel, res, res, res)
            _in_mask = _in_mask.reshape(1, 1, res, res, res)

            in_mask = in_mask + _in_mask
            volume = volume + _feat * _in_mask

            max_in_mask = torch.max(max_in_mask, _in_mask)

            variance = variance + (_feat ** 2) * _in_mask

        eps_threshold = 1e-6
        in_mask[in_mask <= eps_threshold] = 0
        in_mask_expand = in_mask.repeat(1, volume.shape[1], 1, 1, 1)
        non_empty_mask = in_mask_expand > eps_threshold

        volume[non_empty_mask] = volume[non_empty_mask] / in_mask_expand[non_empty_mask]
        volume[~non_empty_mask] = 0

        variance[non_empty_mask] = variance[non_empty_mask] / in_mask_expand[non_empty_mask]
        variance[~non_empty_mask] = 0
        variance = variance - volume ** 2
        volume = torch.cat([volume, variance], dim=1)

        in_mask = in_mask / 1e4
        max_in_mask = max_in_mask / 1e4

        volume = torch.cat([volume, in_mask / len(ref_img), max_in_mask], dim=1)
        volume = self.costreg(volume)
        volume = volume.clamp(-self.clamp_range, self.clamp_range)

        if raw_volume:
            return volume
        else:
            return self.super_resolution(volume)

    def super_resolution(self, volume):
        while len(volume.shape) < 5:
            volume = volume.unsqueeze(0)
        residual_volume = self.sr_net(volume)
        volume = torch.nn.functional.interpolate(volume, scale_factor=2, mode='trilinear')
        volume = volume.repeat(1, int(self.fine_volume_channel // self.coarse_volume_channel), 1, 1, 1)
        volume = volume + residual_volume
        volume = volume.clamp(-self.clamp_range, self.clamp_range)
        return volume

    def forward(self, inputs, ref_img, ref_pose, ref_depth, intrinsic, volume=None):
        inputs = inputs / self.bbox

        if volume is None:
            volume = self.project_volume(ref_img, ref_pose, ref_depth, intrinsic)

        outputs = self.generate_volume_features(inputs, volume)
        return outputs, volume
    
    def get_params(self):
        return list(self.parameters())
