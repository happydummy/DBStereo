from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import tqdm
import argparse

def read_one_img(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            print("File open error:", img_path)
            return img_path
    except:
        print("File does not exist:", img_path)
        return img_path

def read_all_images(input_txt,error_txt):
    with open(input_txt, 'r') as f:
        img_paths = [path.strip() for path in f.readlines()]
    print("Show first image path: ",img_paths[0])
    print("Start Check: it may need hours according to your datasets")

    with ThreadPoolExecutor() as executor:
        target_dir_list = tqdm.tqdm(executor.map(read_one_img, img_paths), total=len(img_paths))
    empty_paths = sorted([x for x in target_dir_list if x is not None])
    with open(error_txt, 'w') as f:
        for path in empty_paths:
            f.write(path + '\n')

def main():
    parser = argparse.ArgumentParser(description="Check image by cv2.imread")
    parser.add_argument('--input_path_file', type=str, help='Path to the txt file containing image paths')
    parser.add_argument('--error_path_file', type=str, help='Path to save the error image files')
    
    args = parser.parse_args()
    input_txt = args.input_path_file
    error_txt = args.error_path_file

    read_all_images(input_txt,error_txt)

if __name__ == "__main__":
    main()

# python check_data.py --input_path_file './image.txt' --error_path_file './image_error.txt'


