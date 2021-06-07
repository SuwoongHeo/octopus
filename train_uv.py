import os
import argparse
import pickle

import tensorflow as tf
# import keras.backend as K
from tensorflow.python.keras import backend as K

from glob import glob
import imageio
import numpy as np
from random import shuffle

from lib.io import openpose_from_file, read_segmentation, write_mesh
from model.octopus2 import Octopus
from dataset import load_base_data, load_data_dict

path = '/ssd2/duc/people_snapshot_public'
out_path = 'results'


def main(weights):
    # segm_files = sorted(glob(os.path.join(segm_dir, '*.png')) + glob(os.path.join(segm_dir, '*.jpg')))
    # img_files = sorted(glob(os.path.join(img_dir, '*.png')) + glob(os.path.join(img_dir, '*.jpg')))
    # pose_files = sorted(glob(os.path.join(pose_dir, '*.json')))

    # if len(segm_files) != len(pose_files) or len(segm_files) == len(pose_files) == 0:
    #     exit('Inconsistent input.')
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)))
    K.set_session(sess)

    model = Octopus(num=1)
    # model.load(weights)
    train_data, val_data = load_base_data(path)
    for epoch in range(10000):
        shuffle(train_data)
        for data_path in train_data:
            input, supervision = load_data_dict(data_path, num_frames=1)
            model.uv_network.fit(input, supervision, batch_size=1)

        for data_path in val_data:
            input, _ = load_data_dict(data_path, num_frames=1)
            output = model.uv_network.predict(input, batch_size=1)
            imageio.imwrite(f'{out_path}/{os.path.split(data_path)[-1]}.jpg', np.clip(output[0], 0., 1.))

    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--weights', '-w',
        default='weights/octopus_weights2.hdf5',
        help='Model weights file (*.hdf5)')

    parser.add_argument(
        '--gpuID', '-g',
        default='0',
        help='GPU ID to use (default : 0), -1 for CPU')

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuID
    # os.makedirs(args.out_dir, exist_ok=True)
    main(args.weights)
