import timm
import torch
import torch.nn as nn

from functools import partial
from stereo.modeling.common.basic_block_2d import BasicConv2d, BasicDeconv2d


class FPNLayer(nn.Module):
    def __init__(self, chan_low, chan_high):
        super().__init__()
        self.deconv = BasicDeconv2d(chan_low, chan_high, kernel_size=4, stride=2, padding=1,
                                    norm_layer=nn.BatchNorm2d,
                                    act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True))

        self.conv = BasicConv2d(chan_high * 2, chan_high*2, kernel_size=3, padding=1,
                                norm_layer=nn.BatchNorm2d,
                                act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True))

    def forward(self, low, high):
        low = self.deconv(low)
        feat = torch.cat([high, low], 1)
        feat = self.conv(feat)
        return feat


class Backbone(nn.Module):
    def __init__(self, backbone='MobileNetv2'):
        super().__init__()
        if backbone == 'MobileNetv2':
            model = timm.create_model('mobilenetv2_100', pretrained=True)
            chans = [16, 24, 32, 96, 160]
            self.output_channels = [64,64,192]
        elif backbone == 'EfficientNetv2':
            model = timm.create_model('efficientnetv2_rw_s', pretrained=True)
            chans = [24, 48, 64, 160, 272]
            self.output_channels = [96,128,320]
        else:
            raise NotImplementedError
        
        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.act1 = model.act1
        self.block0 = model.blocks[0]
        self.block1 = model.blocks[1]
        self.block2 = model.blocks[2]
        self.block3 = model.blocks[3:5]
        self.block4 = model.blocks[5]

        self.deconv32_16 = FPNLayer(chans[4], chans[3])
        self.deconv16_8 = FPNLayer(chans[3]*2, chans[2])
        self.deconv8_4 = FPNLayer(chans[2]*2, chans[1])
        self.conv4 = BasicConv2d(chans[1]*2, self.output_channels[0]-32, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = self.act1(self.bn1(self.conv_stem(x)))
        x2 = self.block0(x1)
        x4 = self.block1(x2)
        x8 = self.block2(x4)
        x16 = self.block3(x8)
        x32 = self.block4(x16)

        x16 = self.deconv32_16(x32, x16)
        x8 = self.deconv16_8(x16, x8)
        x4 = self.deconv8_4(x8, x4)
        x4 = self.conv4(x4)
        return [x4, x8, x16]