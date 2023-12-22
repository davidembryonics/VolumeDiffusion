import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import custom_bwd, custom_fwd
from .encoder import Encoder


class _trunc_exp(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(max=15))

trunc_exp = _trunc_exp.apply


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))
            if l != self.num_layers - 1:
                net.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*net)
    
    def forward(self, x):
        out = self.net(x)
        return out


class NeRFNetwork(nn.Module):
    def __init__(self, opt, device=None,):
        super().__init__()

        self.opt = opt
        self.in_dim = self.opt.fine_volume_channel

        self.sigma_net = MLP(self.in_dim, 4, self.opt.mlp_dim, self.opt.mlp_layer, bias=True)
        self.sigma_net.to(device)

        self.encoder = Encoder(device=device, opt=opt)
        self.encoder.to(device)
        
        self.density_activation = trunc_exp
        
    def forward(self, x, d, ref_img, ref_pose, ref_depth, intrinsic, volume=None):
        with torch.cuda.amp.autocast(enabled=self.opt.fp16):
            enc, volume = self.encoder(x, ref_img, ref_pose, ref_depth, intrinsic, volume=volume)
            h = checkpoint(self.sigma_net, enc, use_reentrant=False)
            sigma = self.density_activation(h[..., 0])
            color = torch.sigmoid(h[..., 1:])
        return {'sigma': sigma, 'color': color}, volume

    def get_params(self, lr0, lr1):
        params = [
            {'params': list(self.encoder.get_params()), 'lr': lr0},
            {'params': list(self.sigma_net.parameters()), 'lr': lr1},
        ]
        return params
