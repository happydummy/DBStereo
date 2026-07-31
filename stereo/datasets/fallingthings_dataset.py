
import os
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from .dataset_template import DatasetTemplate


class FallingThingsDataset(DatasetTemplate):
    def __init__(self, data_info, data_cfg, mode):
        super().__init__(data_info, data_cfg, mode)
        self.return_right_disp = self.data_info.RETURN_RIGHT_DISP
        if hasattr(self.data_info, 'RETURN_SUPER_PIXEL'):
            self.retrun_super_pixel = self.data_info.RETURN_SUPER_PIXEL
        else:
            self.retrun_super_pixel = False

    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item]
        left_path, right_path, left_disp_path = full_paths

        left_img = Image.open(left_path).convert('RGB')
        left_img = np.array(left_img, dtype=np.float32)

        right_img = Image.open(right_path).convert('RGB')
        right_img = np.array(right_img, dtype=np.float32)

        left_depth = Image.open(left_disp_path)
        left_depth = np.array(left_depth, dtype=np.float32)
        left_disp = 460920 / left_depth  # 6cm * 768.2px * 100 = 460920
        occ_mask = np.zeros_like(left_disp, dtype=bool)
        
        sample = {
            'left': left_img,
            'right': right_img,
            'disp': left_disp,
            'occ_mask': occ_mask
        }

        if self.retrun_super_pixel:
            super_pixel_label = Path(self.root).parent.joinpath('SuperPixelLabel/FallingThings', item[0])
            super_pixel_label = str(super_pixel_label)[:-len('.png')] + "_lsc_lbl.png"
            if not os.path.exists(os.path.dirname(super_pixel_label)):
                os.makedirs(os.path.dirname(super_pixel_label))
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

        if self.return_right_disp:
            right_disp_path = left_disp_path.replace('left','right')
            right_depth = Image.open(right_disp_path)
            right_depth = np.array(right_depth, dtype=np.float32)
            right_disp = 460920 / right_depth # 6cm * 768.2px * 100 = 460920
            sample['disp_right'] = right_disp

        sample = self.transform(sample)

        sample['valid'] = sample['disp'] < 512
        sample['index'] = idx
        sample['name'] = left_path

        return sample
    