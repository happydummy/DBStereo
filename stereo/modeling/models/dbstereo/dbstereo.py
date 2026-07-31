import torch.nn as nn
import torch
import torch.nn.functional as F
from stereo.modeling.common.basic_block_2d import BasicConv2d, BasicDeconv2d
from stereo.modeling.cost_volume.cost_volume import build_gwc_volume
from stereo.modeling.disp_pred.disp_regression import disparity_regression
from stereo.modeling.disp_refinement.disp_refinement import context_upsample

from .backbone import Backbone, FPNLayer
from .aggregation import Aggregation, BGA

class DBStereo(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.max_disp = cfgs.MAX_DISP
        self.left_att = cfgs.LEFT_ATT
        aggregation_blocks = [
            block * cfgs.get('AGGREGATION_BLOCKS_LAYERS', 1)
            for block in cfgs.BASE_AGGREGATION_BLOCKS
        ]

        # backbobe
        self.backbone = Backbone(cfgs.get('BACKCONE', 'MobileNetv2'))
        self.stem_2 = nn.Sequential(
            BasicConv2d(3, 16, kernel_size=3, stride=2, padding=1, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU6),
            BasicConv2d(16, 16, kernel_size=3, stride=1, padding=1, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU6)
            )
        self.stem_4 = nn.Sequential(
            BasicConv2d(16, 32, kernel_size=3, stride=2, padding=1, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU6),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU6)
            )

        self.gwc_groups = cfgs.get('GWC_GROUPS', 8)
        self.cost_channels = cfgs.get('COST_CHANNELS', 48)

        in_channels = [48 * self.gwc_groups, 24 * self.gwc_groups, self.cost_channels]

        self.geo_encoder = nn.Sequential(
            BGA(in_channels[0], in_channels[1], stride=1, expanse_ratio=cfgs.EXPANSE_RATIO),
            BGA(in_channels[1], in_channels[2], stride=1, expanse_ratio=cfgs.EXPANSE_RATIO)
        )

        self.geo_decoder = Aggregation(in_channels=self.cost_channels,
                                    left_att=self.left_att,
                                    blocks=aggregation_blocks,
                                    expanse_ratio=cfgs.EXPANSE_RATIO,
                                    backbone_channels=self.backbone.output_channels)

        # disp refine
        self.refine_1 = nn.Sequential(
            BasicConv2d(self.backbone.output_channels[0], 32, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU6),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU6))

        self.refine_2 = FPNLayer(32, 16)

        self.refine_3 = BasicDeconv2d(16*2, 9, kernel_size=4, stride=2, padding=1)



    def forward(self, data):
        image1 = data['left']
        image2 = data['right']

        features_left = self.backbone(image1)
        features_right = self.backbone(image2)

        stem_2x = self.stem_2(image1)
        stem_4x = self.stem_4(stem_2x)
        stem_2y = self.stem_2(image2)
        stem_4y = self.stem_4(stem_2y)

        features_left[0] = torch.cat((features_left[0], stem_4x), 1)
        features_right[0] = torch.cat((features_right[0], stem_4y), 1)

        gwc_volume = build_gwc_volume(features_left[0], features_right[0], self.max_disp // 4, self.gwc_groups)
        b, c, disp, h, w = gwc_volume.shape
        gwc_volume = gwc_volume.view(b, -1, h, w)  # [bz, c*disp, H/4, W/4]
        
        gwc_volume = self.geo_encoder(gwc_volume)  

        encoding_volume = self.geo_decoder(gwc_volume, features_left)  # [bz, 1, max_disp/4, H/4, W/4]
        
        squeezed_encoding = encoding_volume[0].reshape(encoding_volume[0].size(0), -1, encoding_volume[0].size(2), encoding_volume[0].size(3))  # [bz, max_disp/4, H/4, W/4]
        prob = F.softmax(squeezed_encoding, dim=1) 
       
        init_disp = disparity_regression(prob, self.max_disp // 4)  

        xspx = self.refine_1(features_left[0])
        xspx = self.refine_2(xspx, stem_2x)
        xspx = self.refine_3(xspx)
        spx_pred = F.softmax(xspx, 1) 
        disp_pred = context_upsample(init_disp * 4., spx_pred.float()).unsqueeze(1) 

        result = {'disp_pred': disp_pred}

        if self.training:
            disp_4 = F.interpolate(init_disp, image1.shape[2:], mode='bilinear', align_corners=False)
            disp_4 *= 4
            result['disp_4'] = disp_4
        return result

    def get_loss(self, model_pred, input_data):
        disp_gt = input_data["disp"]  # [bz, h, w]
        disp_gt = disp_gt.unsqueeze(1)  # [bz, 1, h, w]
        mask = (disp_gt < self.max_disp) & (disp_gt > 0)  # [bz, 1, h, w]
        disp_pred = model_pred['disp_pred']
        loss = 1.0 * F.smooth_l1_loss(disp_pred[mask], disp_gt[mask], reduction='mean')

        disp_4 = model_pred['disp_4']
        loss += 0.3 * F.smooth_l1_loss(disp_4[mask], disp_gt[mask], reduction='mean')
        loss_info = {'scalar/train/loss_disp': loss.item()
                    }

        return loss, loss_info
