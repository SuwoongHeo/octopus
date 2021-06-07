import tensorflow as tf
import os
from random import shuffle
from glob import glob
import numpy as np
import imageio
from skimage import transform

path = '/ssd2/duc/people_snapshot_public'


def load_base_data(path):
    data = os.listdir(path)
    data = [os.path.join(path, p) for p in data]
    split = int(.8 * len(data))
    return data[:split], data[split:]


def load_data_dict(path, num_frames=8):
    img_list = sorted(glob(os.path.join(path, 'frames/*.jpg')))
    indices = np.random.permutation(len(img_list))[:num_frames]
    img_names = [os.path.join(path, f'frames/frame{i}.jpg') for i in indices]
    seg_names = [os.path.join(path, f'segmentations/frame{i}_parsing.png') for i in indices]
    images = [np.array(imageio.imread(img_name), 'float32') / 255. for img_name in img_names]
    segs = [np.array(imageio.imread(seg_name), 'float32') / 255. for seg_name in seg_names]

    gt = glob(os.path.join(path, '*.jpg'))[0]
    gt = transform.resize(np.array(imageio.imread(gt), 'float32'), images[0].shape[:-1])[None] / 255.

    data_dict = {}
    for i, img in enumerate(images):
        data_dict[f'image_{i}'] = img[None]
        data_dict[f'segmentation_{i}'] = segs[i][None]

    return data_dict, {'uv': gt}
