import os
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from stereo.datasets.dataset_utils.readpfm import readpfm
from .dataset_template import DatasetTemplate


class VirtualKitti2Dataset(DatasetTemplate):
    def __init__(self, data_info, data_cfg, mode):
        super().__init__(data_info, data_cfg, mode)
        self.return_right_disp = self.data_info.RETURN_RIGHT_DISP
        if hasattr(self.data_info, 'RETURN_SUPER_PIXEL'):
            self.retrun_super_pixel = self.data_info.RETURN_SUPER_PIXEL
        else:
            self.retrun_super_pixel = False

    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item[0:4]]
        # left_img_path, right_img_path, disp_img_path, disp_img_right_path = full_paths
        left_img_path, right_img_path, disp_img_path = full_paths
        left_img = Image.open(left_img_path).convert('RGB')
        left_img = np.array(left_img, dtype=np.float32)
        right_img = Image.open(right_img_path).convert('RGB')
        right_img = np.array(right_img, dtype=np.float32)
        disp_img = get_disp(disp_img_path).astype(np.float32)
        assert not np.isnan(disp_img).any(), 'disp_img has nan'
        occ_mask = np.zeros_like(disp_img, dtype=bool)

        sample = {
            'left': left_img,  # [H, W, 3]
            'right': right_img,  # [H, W, 3]
            'disp': disp_img,  # [H, W]
            # 'occ_mask': occ_mask
        }

        if self.retrun_super_pixel:
            super_pixel_label = Path(self.root).parent.joinpath('SuperPixelLabel/VirtualKitti2', item[0])
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

        if self.return_right_disp:
            disp_img_right = get_disp(disp_img_right_path).astype(np.float32)
            sample['disp_right'] = disp_img_right
            assert not np.isnan(disp_img_right).any(), 'disp_img_right has nan'

        sample = self.transform(sample)

        # sample['valid'] = sample['disp'] < 512
        sample['index'] = idx
        sample['name'] = left_img_path
        return sample


def get_disp(file_path, checkinvalid=True):
    if '.png' in file_path:
        depth = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        invalid = depth >= 65535
        num_invalid = depth[invalid].shape[0]
        depth = depth / 100.0
    else:
        raise TypeError('only support png and npy format, invalid type found: {}'.format(file_path))

    f = 725.0087
    b = 0.532725 # meter

    disp = b * f / (depth + 1e-5)
    if checkinvalid:
        disp[invalid] = 0
    return disp