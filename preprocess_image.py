import os
import cv2
import glob
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt



def get_image_list(input_spec, exp=None):
    if os.path.isdir(input_spec):
        file_list = [
            os.path.join(input_spec, fname)
            for fname in os.listdir(input_spec)
            if os.path.isfile(os.path.join(input_spec, fname))
        ]
    elif os.path.isfile(input_spec):
        file_list = [input_spec]
    else:
        file_list = glob.glob(input_spec)
    if exp!=None:
        for idx_, file_ in enumerate(file_list):
            if exp not in file_.split('/')[-1] :
                file_list.pop(idx_)
    return file_list


def main(args):
    target_size = (args.img_size, args.img_size, 3)
    y_const = 920  # constraint along with height
    os.makedirs(args.out_path, exist_ok=True)
    image_list = get_image_list(args.input_path, ['jpg', 'png'])
    for image_path in image_list:
        rgb_image = cv2.imread(image_path)
        image_name = image_path.split('/')[-1].split('.')[0]
        with open(os.path.join(args.bbox_path, f'{image_name}_bbox.json'), 'r') as f:
            bbox = json.load(f)['bbox'][0]

        bbox[2] -= bbox[0]
        bbox[3] -= bbox[1]

        x, y, w, h = [int(v) for v in bbox[:4]]
        ratio = y_const / h if h > y_const else 1
        rgb_crop = cv2.resize(rgb_image[y:y + h, x:x + w, :], (int(w * ratio), int(h * ratio)))

        top, bottom = np.floor((target_size[0] - rgb_crop.shape[0]) / 2).astype(np.int32), \
                      np.ceil((target_size[0] - rgb_crop.shape[0]) / 2).astype(np.int32)
        left, right = np.floor((target_size[1] - rgb_crop.shape[1]) / 2).astype(np.int32), \
                      np.ceil((target_size[1] - rgb_crop.shape[1]) / 2).astype(np.int32)

        rgb_out = cv2.copyMakeBorder(rgb_crop, top, bottom, left, right, cv2.BORDER_CONSTANT)

        cv2.imwrite(os.path.join(args.out_path, f"{image_name}_{args.img_size}x{args.img_size}.{image_path.split('.')[-1]}"), rgb_out)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_path', default="/ssd2/swheo/dev/HumanRecon_/octopus/data/Sample_BEJ/4view_nocalib/segmentations",
        type=str, help="Path to single image wit formats (jpg, png, bmp, etc.), or directory")
    parser.add_argument(
        '--bbox_path', default="/ssd2/swheo/dev/HumanRecon_/octopus/data/Sample_BEJ/4view_nocalib/segmentations",
        type=str, help="Bbox path")

    parser.add_argument("--img_size", default=1080, type=int, help="Enable the black background")
    parser.add_argument("--out_path", default="/ssd2/swheo/dev/octopus/data/Sample_BEJ/4view/images_1080",
                        help="Path to write output images")
    args = parser.parse_args()
    main(args)