import os
import argparse
import tensorflow as tf
# import keras.backend as K
from tensorflow.python.keras import backend as K

from glob import glob

from lib.io import openpose_from_file, read_segmentation, write_mesh
from model.octopus import Octopus

def main(weights, name, segm_dir, pose_dir, out_dir, opt_pose_steps, opt_shape_steps):
    segm_files = sorted(glob(os.path.join(segm_dir, '*.png')))
    pose_files = sorted(glob(os.path.join(pose_dir, '*.json')))

    if len(segm_files) != len(pose_files) or len(segm_files) == len(pose_files) == 0:
        exit('Inconsistent input.')
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)))
    K.set_session(sess)

    model = Octopus(num=len(segm_files))
    model.load(weights)

    segmentations = [read_segmentation(f) for f in segm_files]

    joints_2d, face_2d = [], []
    for f in pose_files:
        j, f = openpose_from_file(f)

        assert(len(j) == 25)
        assert(len(f) == 70)

        joints_2d.append(j)
        face_2d.append(f)

    if opt_pose_steps:
        print('Optimizing for pose...')
        model.opt_pose(segmentations, joints_2d, opt_steps=opt_pose_steps)

    if opt_shape_steps:
        print('Optimizing for shape...')
        model.opt_shape(segmentations, joints_2d, face_2d, opt_steps=opt_shape_steps)

    print('Estimating shape...')
    pred = model.predict(segmentations, joints_2d)

    write_mesh('{}/{}.obj'.format(out_dir, name), pred['vertices'][0], pred['faces'])

    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'name',
        type=str, default='none',
        help="Sample name")

    parser.add_argument(
        'segm_dir',
        type=str, default='data/sample/segmentations',
        help="Segmentation images directory")

    parser.add_argument(
        'pose_dir',
        type=str, default='data/sample/keypoints',
        help="2D pose keypoints directory")

    parser.add_argument(
        '--opt_steps_pose', '-p', default=5, type=int,
        help="Optimization steps pose")

    parser.add_argument(
        '--opt_steps_shape', '-s', default=15, type=int,
        help="Optimization steps")

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
        default='0',
        help='GPU ID to use (default : 0), -1 for CPU')

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuID
    main(args.weights, args.name, args.segm_dir, args.pose_dir, args.out_dir, args.opt_steps_pose, args.opt_steps_shape)
