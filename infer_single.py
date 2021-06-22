import os
import argparse
import tensorflow as tf
import pickle as pkl
# import keras.backend as K
from tensorflow.python.keras import backend as K

from glob import glob

from lib.io import openpose_from_file, read_segmentation, write_mesh
from model.octopus import Octopus

def main(weights, name, segm_dir, pose_dir, out_dir, opt_pose_steps, opt_shape_steps):
    segm_files = sorted(glob(os.path.join(segm_dir, '*.png')) + glob(os.path.join(segm_dir, '*.jpg')))
    pose_files = sorted(glob(os.path.join(pose_dir, '*.json')))

    if len(segm_files) != len(pose_files) or len(segm_files) == len(pose_files) == 0:
        print(f"segm_dir : {segm_dir}, pose_dir: {pose_dir}")
        print(f"segm_files : {len(segm_files)}, pose_files : {len(pose_files)}")
        exit('Inconsistent input.')
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)))
    K.set_session(sess)

    model = Octopus(num=len(segm_files))
    model.load(weights)

    # Test reset weight
    model.set_weights()

    segmentations = [read_segmentation(f) for f in segm_files]

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

    print('Estimating shape...')
    import time
    pred = model.predict(segmentations, joints_2d)
    t = time.time()
    pred = model.predict(segmentations, joints_2d)
    print('elapsed {}'.format(time.time() - t))

    write_mesh('{}/{}.obj'.format(out_dir, name), pred['vertices_tposed'], pred['faces'])
    for idx in range(pred['vertices'].shape[0]):
        write_mesh('{}/{}_view{}.obj'.format(out_dir, name, idx), pred['vertices'][idx], pred['faces'])

    tposed = pred['vertices_tposed'] - pred['offsets']
    write_mesh('{}/{}_wo_offsets.obj'.format(out_dir, name), tposed, pred['faces'])
    with open('./assets/neutral_smpl_hres.pkl', 'rb') as f:
        hres_smpl = pkl.load(f, encoding='latin-1')
    write_mesh('{}/{}_hres.obj'.format(out_dir, name), hres_smpl['ss']*pred['vertices_tposed'], hres_smpl['f'])
    with open('{}/{}_pred.pkl'.format(out_dir, name), 'wb') as f:
        pkl.dump(pred, f)

    print('Done.')


if __name__ == '__main__':
    #BEJ8_2 data/Sample_BEJ/chosen/segmentations data/Sample_BEJ/chosen/keypoints
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--name',
        type=str, default='none',
        help="Sample name")

    parser.add_argument(
        '--segm_dir',
        type=str, default='data/sample/segmentations',
        help="Segmentation images directory")

    parser.add_argument(
        '--pose_dir',
        type=str, default='data/sample/keypoints',
        help="2D pose keypoints directory")

    parser.add_argument(
        '--opt_steps_pose', '-p', default=20, type=int,
        help="Optimization steps pose")

    parser.add_argument(
        '--opt_steps_shape', '-s', default=5, type=int,
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
        default='1',
        help='GPU ID to use (default : 0), -1 for CPU')

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuID
    main(args.weights, args.name, args.segm_dir, args.pose_dir, args.out_dir, args.opt_steps_pose, args.opt_steps_shape)
