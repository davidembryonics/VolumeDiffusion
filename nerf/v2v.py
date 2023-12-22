import torch.nn as nn
import torch.nn.functional as F


class Res3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Res3DBlock, self).__init__()
        self.res_branch = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True),
            nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_planes)
        )

        if in_planes == out_planes:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(out_planes)
            )
    
    def forward(self, x):
        res = self.res_branch(x)
        skip = self.skip_con(x)
        return F.relu(res + skip, True)

    
class Pool3DBlock(nn.Module):
    def __init__(self, pool_size):
        super(Pool3DBlock, self).__init__()
        self.pool_size = pool_size
    
    def forward(self, x):
        return F.max_pool3d(x, kernel_size=self.pool_size, stride=self.pool_size)
    

class Upsample3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Upsample3DBlock, self).__init__()
        assert(kernel_size == 2)
        assert(stride == 2)
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0, output_padding=0),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)
    

class EncoderDecorder(nn.Module):
    def __init__(self, base_ch=32, ch_mult=(1,2,4)):
        super(EncoderDecorder, self).__init__()

        self.base_ch = base_ch
        self.ch_mult = ch_mult

        chs = [(self.base_ch * m) for m in self.ch_mult]
        assert len(chs) == 3

        self.encoder_pool1 = Pool3DBlock(2)
        self.encoder_res1 = nn.Sequential(Res3DBlock(chs[0], chs[1]), Res3DBlock(chs[1], chs[1]))
        self.encoder_pool2 = Pool3DBlock(2)
        self.encoder_res2 = nn.Sequential(Res3DBlock(chs[1], chs[2]), Res3DBlock(chs[2], chs[2]))

        self.mid_res = nn.Sequential(Res3DBlock(chs[2], chs[2]), Res3DBlock(chs[2], chs[2]))

        self.decoder_res2 = nn.Sequential(Res3DBlock(chs[2], chs[2]), Res3DBlock(chs[2], chs[1]))
        self.decoder_upsample2 = Upsample3DBlock(chs[1], chs[1], 2, 2)
        self.decoder_res1 = nn.Sequential(Res3DBlock(chs[1], chs[1]), Res3DBlock(chs[1], chs[0]))
        self.decoder_upsample1 = Upsample3DBlock(chs[0], chs[0], 2, 2)

        self.skip_res1 = nn.Sequential(Res3DBlock(chs[0], chs[0]), Res3DBlock(chs[0], chs[0]))
        self.skip_res2 = nn.Sequential(Res3DBlock(chs[1], chs[1]), Res3DBlock(chs[1], chs[1]))

    def forward(self, x):
        skip_x1 = self.skip_res1(x)
        x = self.encoder_pool1(x)
        x = self.encoder_res1(x)

        skip_x2 = self.skip_res2(x)
        x = self.encoder_pool2(x)
        x = self.encoder_res2(x)

        x = self.mid_res(x)

        x = self.decoder_res2(x)
        x = self.decoder_upsample2(x)
        x = x + skip_x2

        x = self.decoder_res1(x)
        x = self.decoder_upsample1(x)
        x = x + skip_x1

        return x


class V2VNet(nn.Module):
    def __init__(self, input_channels, output_channels, base_ch=32, ch_mult=(1,2,4)):
        super(V2VNet, self).__init__()

        self.base_ch = base_ch
        self.ch_mult = ch_mult

        self.front_layers = nn.Sequential(
            Res3DBlock(input_channels, self.base_ch * self.ch_mult[0]),
        )

        self.encoder_decoder = EncoderDecorder(self.base_ch, self.ch_mult)

        self.output_layer = nn.Conv3d(self.base_ch * self.ch_mult[0], output_channels, kernel_size=1, stride=1, padding=0)

        self._initialize_weights()

    def forward(self, x):
        x = self.front_layers(x)
        x = self.encoder_decoder(x)
        x = self.output_layer(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)


class EncoderDecorderSR(nn.Module):
    def __init__(self, base_ch=32, ch_mult=(1,1)):
        super(EncoderDecorderSR, self).__init__()

        self.base_ch = base_ch
        self.ch_mult = ch_mult

        chs = [(self.base_ch * m) for m in self.ch_mult]
        assert len(chs) == 2

        self.decoder_1 = nn.Sequential(Res3DBlock(chs[0], chs[0]), Res3DBlock(chs[0], chs[0]), Res3DBlock(chs[0], chs[0]))
        self.decoder_up = Upsample3DBlock(chs[0], chs[1], 2, 2)
        self.decoder_2 = nn.Sequential(Res3DBlock(chs[1], chs[1]), Res3DBlock(chs[1], chs[1]), Res3DBlock(chs[1], chs[1]))

    def forward(self, x):
        skip = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)

        x = self.decoder_1(x)
        x = self.decoder_up(x)
        x = self.decoder_2(x)
        x = x + skip

        return x


class V2VNetSR(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(V2VNetSR, self).__init__()

        self.base_ch = 64
        self.ch_mult = (1, 1)

        self.front_layers = nn.Sequential(
            Res3DBlock(input_channels, self.base_ch * self.ch_mult[0]),
        )

        self.encoder_decoder = EncoderDecorderSR(self.base_ch, self.ch_mult)

        self.output_layer = nn.Conv3d(self.base_ch * self.ch_mult[0], output_channels, kernel_size=1, stride=1, padding=0)

        self._initialize_weights()

    def forward(self, x, dummy=None):
        x = self.front_layers(x)
        x = self.encoder_decoder(x)
        x = self.output_layer(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
