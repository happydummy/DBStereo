import numpy as np
from PIL import Image
import argparse
import sys
sys.path.insert(0, './')
from stereo.datasets.dataset_utils.readpfm import readpfm
from stereo.utils.disp_color import disp_to_color


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--disp_path', type=str, default=None)
    parser.add_argument('--savename', type=str, default=None)
    parser.add_argument('--type', type=str, default='pfm', help='file type of disparity')
    args = parser.parse_args()

    if args.type == 'png':
        disp = Image.open(args.disp_path)
    elif args.type == 'pfm':
        disp = readpfm(args.disp_path)[0].astype(np.float32)
    else:
        print("Type Undefined!")

    disp = np.ascontiguousarray(disp, dtype=np.float32)

    disp_color = disp_to_color(disp, max_disp=192)
    disp_color = disp_color.astype('uint8')
    disp_color = Image.fromarray(disp_color)
    disp_color.save(args.savename)
