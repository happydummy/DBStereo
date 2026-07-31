
import os
import numpy as np
import h5py
import cv2
from PIL import Image
from pathlib import Path
from .dataset_template import DatasetTemplate


class SpringDataset(DatasetTemplate):
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
        left_path, right_path, disp_path = full_paths

        left_img = Image.open(left_path).convert('RGB')
        left_img = np.array(left_img, dtype=np.float32)

        right_img = Image.open(right_path).convert('RGB')
        right_img = np.array(right_img, dtype=np.float32)

        disp_img = self._readDsp5Disp(disp_path)
        disp_img = np.ascontiguousarray(disp_img, dtype=np.float32)[::2, ::2]
        occ_mask = np.zeros_like(disp_img, dtype=bool)

        sample = {
            'left': left_img,
            'right': right_img,
            'disp': disp_img,
            'occ_mask': occ_mask
        }

        if self.retrun_super_pixel:
            super_pixel_label = Path(self.root).parent.joinpath('SuperPixelLabel/Spring', item[0])
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
            disp_right_path = disp_path.replace('left', 'right')
            right_disp_img = self._readDsp5Disp(disp_right_path)
            right_disp_img = np.ascontiguousarray(right_disp_img, dtype=np.float32)[::2, ::2]
            sample['disp_right'] = right_disp_img

        sample = self.transform(sample)

        sample['valid'] = sample['disp'] < 512
        sample['index'] = idx
        sample['name'] = left_path

        return sample
    
    def _readDsp5Disp(self, filename):
        with h5py.File(filename, "r") as f:
            if "disparity" not in f.keys():
                raise IOError(f"File {filename} does not have a 'disparity' key. Is this a valid dsp5 file?")
            return f["disparity"][()]