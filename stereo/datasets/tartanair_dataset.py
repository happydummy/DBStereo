import os
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from .dataset_template import DatasetTemplate


class TartanAirDataset(DatasetTemplate):
    def __init__(self, data_info, data_cfg, mode):
        super().__init__(data_info, data_cfg, mode)
        if hasattr(self.data_info, 'RETURN_SUPER_PIXEL'):
            self.retrun_super_pixel = self.data_info.RETURN_SUPER_PIXEL
        else:
            self.retrun_super_pixel = False

    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item]
        left_img_path, right_img_path, left_depth_path = full_paths
        # image
        left_img = np.array(Image.open(left_img_path).convert('RGB'), dtype=np.float32)
        right_img = np.array(Image.open(right_img_path).convert('RGB'), dtype=np.float32)
        # disp
        depth = np.load(left_depth_path)
        disparity = 80.0 / depth
        occ_mask = np.zeros_like(disparity, dtype=bool)

        sample = {
            'left': left_img,
            'right': right_img,
            'disp': disparity,
            'occ_mask': occ_mask
        }

        if self.retrun_super_pixel:
            super_pixel_label = Path(self.root).parent.joinpath('SuperPixelLabel/TartanAir', item[0])
            super_pixel_label = str(super_pixel_label)[:-len('.png')] + "_lsc_lbl.png"
            if not os.path.exists(os.path.dirname(super_pixel_label)):
                os.makedirs(os.path.dirname(super_pixel_label), exist_ok=True)
            if not os.path.exists(super_pixel_label):
                img = cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR)
                lsc = cv2.ximgproc.createSuperpixelLSC(img, region_size=10, ratio=0.075)
                lsc.iterate(20)
                label = lsc.getLabels()
                cv2.imwrite(super_pixel_label, label.astype(np.uint16))
            super_pixel_label = cv2.imread(super_pixel_label, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if super_pixel_label is None:
                img = cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR)
                lsc = cv2.ximgproc.createSuperpixelLSC(img, region_size=10, ratio=0.075)
                lsc.iterate(20)
                label = lsc.getLabels()
                super_pixel_label = label.astype(np.int32)
            else:
                super_pixel_label = super_pixel_label.astype(np.int32)
            sample['super_pixel_label'] = super_pixel_label

        sample = self.transform(sample)

        sample['valid'] = sample['disp'] < 512
        sample['index'] = idx
        sample['name'] = left_img_path
        return sample
