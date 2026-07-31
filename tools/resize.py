import cv2
import tqdm
import os
from concurrent.futures import ThreadPoolExecutor
import argparse

def resize_image(input_path, input_path_root,output_path_root,target_size=(768, 384)):
    try:
        img = cv2.imread(input_path)
        if img is None:
            print("File open error:", input_path)
            return
    except:
        print("File does not exist:", input_path)
        return

    output_path = input_path.replace(input_path_root, output_path_root)
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    original_height, original_width = img.shape[:2]

    # Calculate the scaling ratio
    scale = max(target_size[1] / original_height, target_size[0] / original_width)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Ensure that new_width or new_height equals the target size
    if new_width < target_size[0]:
        new_width = target_size[0]
        new_height = int(original_height * (new_width / original_width))  # Recalculate the height
    elif new_height < target_size[1]:
        new_height = target_size[1]
        new_width = int(original_width * (new_height / original_height))  # Recalculate the width
    else:
        pass

    resized_img = cv2.resize(img, (new_width, new_height))

    # save images
    cv2.imwrite(output_path, resized_img)

def resize_all_images(all_img_txt,input_path_root,output_path_root):
    # Use ThreadPoolExecutor to process files in parallel 
    with open(all_img_txt, 'r') as f:
        input_files = sorted([file.strip() for file in f.readlines()])

    with ThreadPoolExecutor() as executor:
        with tqdm.tqdm(total=len(input_files), desc="Resizing images", unit="img") as pbar:
            for _ in executor.map(lambda img_path: resize_image(img_path, input_path_root,output_path_root),input_files):
                pbar.update(1)

def main():
    parser = argparse.ArgumentParser(description="Resize all image paths from a txt file.")
    parser.add_argument('--input_path_file', type=str, help='Path to the txt file containing image paths')
    parser.add_argument('--input_path_root', type=str, help='Root directory containing input images')
    parser.add_argument('--output_path_root', type=str, help='Root directory to save output paths')
    
    args = parser.parse_args()
    all_img_txt = args.input_path_file
    input_path_root = args.input_path_root
    output_path_root = args.output_path_root

    resize_all_images(all_img_txt,input_path_root,output_path_root)

if __name__ == "__main__":
    main()

# example
# python resize.py --input_path_file ./xxx.txt --input_path_root "/path/to/imagenet21k/" --output_path_root "./imagenet21k/"


