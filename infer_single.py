import os
import argparse
import tensorflow as tf
# import keras.backend as K
from tensorflow.python.keras import backend as K

from glob import glob
import imageio
import numpy as np

from lib.io import openpose_from_file, read_segmentation, write_mesh
from model.octopus import Octopus


def main(weights, name, img_dir, segm_dir, pose_dir, out_dir, opt_pose_steps, opt_shape_steps, opt_texture_steps):
    segm_files = sorted(glob(os.path.join(segm_dir, '*.png')) + glob(os.path.join(segm_dir, '*.jpg')))
    img_files = sorted(glob(os.path.join(img_dir, '*.png')) + glob(os.path.join(img_dir, '*.jpg')))
    pose_files = sorted(glob(os.path.join(pose_dir, '*.json')))

    if len(segm_files) != len(pose_files) or len(segm_files) == len(pose_files) == 0:
        exit('Inconsistent input.')
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)))
    K.set_session(sess)

    model = Octopus(num=len(segm_files))
    model.load(weights)

    segmentations = [read_segmentation(f) for f in segm_files]
    images = [np.array(imageio.imread(f), 'float32') / 255. for f in img_files]

    joints_2d, face_2d = [], []
    j_2d_test, f_2d_test = [], []
    for f in pose_files:
        j, f, j_o, f_o = openpose_from_file(f)

        assert(len(j) == 25)
        assert(len(f) == 70)

        joints_2d.append(j)
        face_2d.append(f)
        j_2d_test.append(j_o)
        f_2d_test.append(f_o)

    if opt_pose_steps:
        print('Optimizing for pose...')
        model.opt_pose(segmentations, joints_2d, opt_steps=opt_pose_steps)

    if opt_shape_steps:
        print('Optimizing for shape...')
        model.opt_shape(segmentations, joints_2d, face_2d, opt_steps=opt_shape_steps)

    if opt_texture_steps:
        print('Optimizing for texture...')
        model.opt_texture(segmentations, joints_2d, images, opt_texture_steps)

    print('Estimating shape...')
    pred = model.predict(segmentations, joints_2d)

    write_mesh('{}/{}.obj'.format(out_dir, name), pred['vertices'][0], pred['faces'])
    for i in range(len(pred['rendered_color'])):
        imageio.imwrite(f'{out_dir}/{name}_{i}_color.png', pred['rendered_color'][i, 0])

    # imageio.imwrite(f'{out_dir}/{name}.png', model.uv.numpy())

    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'name',
        type=str, default='none',
        help="Sample name")

    parser.add_argument(
        'img_dir',
        type=str, default='data/sample/frames',
        help="RGB images directory")

    parser.add_argument(
        'segm_dir',
        type=str, default='data/sample/segmentations',
        help="Segmentation images directory")

    parser.add_argument(
        'pose_dir',
        type=str, default='data/sample/keypoints',
        help="2D pose keypoints directory")

    parser.add_argument(
        '--opt_steps_pose', '-p', default=20, type=int,
        help="Optimization steps pose")

    parser.add_argument(
        '--opt_steps_shape', '-s', default=5, type=int,
        help="Optimization steps")

    parser.add_argument(
        '--opt_steps_texture', '-t', default=500, type=int,
        help="Optimization steps for texture")

    parser.add_argument(
        '--out_dir', '-od',
        default='out',
        help='Output directory')

    parser.add_argument(
        '--weights', '-w',
        default='weights/octopus_weights2.hdf5',
        help='Model weights file (*.hdf5)')

    parser.add_argument(
        '--gpuID', '-g',
        default='6',
        help='GPU ID to use (default : 0), -1 for CPU')

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuID
    main(args.weights, args.name, args.img_dir, args.segm_dir, args.pose_dir, args.out_dir, args.opt_steps_pose,
         args.opt_steps_shape, args.opt_steps_texture)
