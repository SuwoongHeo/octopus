# Code partly taken from https://github.com/akanazawa/hmr.
# The following license applies:
#
#
# MIT License
#
# This code base itself is MIT, but please follow the license for SMPL, MoSh data,
# and the respective dataset.
#
# Copyright (c) 2018 akanazawa
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""
Tensorflow SMPL implementation as batch.
Specify joint types:
'coco': Returns COCO+ 19 joints
'lsp': Returns H3.6M-LSP 14 joints
Note: To get original smpl joints, use self.J_transformed
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np

import tensorflow as tf
from .batch_lbs import batch_rodrigues, batch_global_rigid_transformation

import pickle


# There are chumpy variables so convert them to numpy.
def undo_chumpy(x):
    return x if isinstance(x, np.ndarray) else x.r


def sparse_to_tensor(x, dtype=tf.float32):
    coo = x.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, tf.convert_to_tensor(value=coo.data, dtype=dtype), coo.shape)

class SMPL(object):
# class SMPL(tf.keras.layers.Layer):
    def __init__(self, pkl_path, theta_in_rodrigues=True, theta_is_perfect_rotmtx=True, dtype=tf.float32):
        # super(SMPL, self).__init__(name='smpl_main')
        """
        pkl_path is the path to a SMPL model
        """
        # -- Load SMPL params --
        with open(pkl_path, 'rb') as f:
            dd = pickle.load(f, encoding='iso-8859-1')
        # Mean template vertices
        self.v_template = tf.constant(
            undo_chumpy(dd['v_template']),
            dtype=dtype)
        # Size of mesh [Number of vertices, 3]
        self.size = [self.v_template.shape[0], 3]
        self.num_betas = dd['shapedirs'].shape[-1]
        # Shape blend shape basis: 6980 x 3 x 10
        # reshaped to 6980*30 x 10, transposed to 10x6980*3
        shapedir = np.reshape(
            undo_chumpy(dd['shapedirs']), [-1, self.num_betas]).T
        self.shapedirs = tf.constant(
            shapedir, dtype=dtype)

        # Regressor for joint locations given shape - 6890 x 24
        self.J_regressor = sparse_to_tensor(dd['J_regressor'], dtype=dtype)

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*30 x 207
        num_pose_basis = dd['posedirs'].shape[-1]
        # 207 x 20670
        posedirs = np.reshape(
            undo_chumpy(dd['posedirs']), [-1, num_pose_basis]).T
        self.posedirs = tf.constant(
            posedirs, dtype=dtype,)

        # indices of parents for each joints
        self.parents = dd['kintree_table'][0].astype(np.int32)

        # LBS weights
        self._weights = tf.constant(
            undo_chumpy(dd['weights']),
            dtype=dtype)

        # expect theta in rodrigues form
        self.theta_in_rodrigues = theta_in_rodrigues

        # if in matrix form, is it already rotmax?
        self.theta_is_perfect_rotmtx = theta_is_perfect_rotmtx

    def __call__(self, inputs):
        """
        Obtain SMPL with shape (beta) & pose (theta) inputs.
        Theta includes the global rotation.
        Args:
          beta: N x 10
          theta: N x 72 (with 3-D axis-angle rep)

        Updates:
        self.J_transformed: N x 24 x 3 joint location after shaping
                 & posing with beta and theta
        Returns:
          - joints: N x 19 or 14 x 3 joint locations depending on joint_type
        If get_skin is True, also returns
          - Verts: N x 6980 x 3
        """
        theta, beta, trans, v_personal = inputs
        name = None
        with tf.compat.v1.name_scope(name, "smpl_main", [beta, theta, trans, v_personal]):
            num_batch = tf.shape(beta)[0]

            # 1. Add shape blend shapes
            # (N x 10) x (10 x 6890*3) = N x 6890 x 3
            v_shaped_scaled = tf.reshape(
                tf.matmul(beta, self.shapedirs, name='shape_bs'),
                [-1, self.size[0], self.size[1]]) + self.v_template

            body_height = (v_shaped_scaled[:, 2802, 1] + v_shaped_scaled[:, 6262, 1]) - (v_shaped_scaled[:, 2237, 1] + v_shaped_scaled[:, 6728, 1])
            scale = tf.reshape(1.66 / body_height, (-1, 1, 1))

            v_shaped = scale * v_shaped_scaled
            v_shaped_personal = v_shaped + v_personal

            # 2. Infer shape-dependent joint locations.
            Jx = tf.transpose(a=tf.sparse.sparse_dense_matmul(self.J_regressor, tf.transpose(a=v_shaped_scaled[:, :, 0])))
            Jy = tf.transpose(a=tf.sparse.sparse_dense_matmul(self.J_regressor, tf.transpose(a=v_shaped_scaled[:, :, 1])))
            Jz = tf.transpose(a=tf.sparse.sparse_dense_matmul(self.J_regressor, tf.transpose(a=v_shaped_scaled[:, :, 2])))
            J = scale * tf.stack([Jx, Jy, Jz], axis=2)

            # 3. Add pose blend shapes
            # N x 24 x 3 x 3
            if self.theta_in_rodrigues:
                Rs = tf.reshape(
                    batch_rodrigues(tf.reshape(theta, [-1, 3])), [-1, 24, 3, 3])
            else:
                if self.theta_is_perfect_rotmtx:
                    Rs = theta
                else:
                    # s, u, v = tf.svd(theta)
                    s, u, v = tf.linalg.svd(theta)
                    Rs = tf.matmul(u, tf.transpose(a=v, perm=[0, 1, 3, 2]))

            with tf.compat.v1.name_scope("lrotmin"):
                # Ignore global rotation.
                pose_feature = tf.reshape(Rs[:, 1:, :, :] - tf.eye(3), [-1, 207])

            # (N x 207) x (207, 20670) -> N x 6890 x 3
            v_posed = tf.reshape(
                tf.matmul(pose_feature, self.posedirs),
                [-1, self.size[0], self.size[1]]) + v_shaped_personal

            #4. Get the global joint location
            J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents)
            J_transformed += tf.expand_dims(trans, axis=1)

            # 5. Do skinning:
            # W is N x 6890 x 24
            W = tf.reshape(
                tf.tile(self._weights, [num_batch, 1]), [num_batch, -1, 24])
            # (N x 6890 x 24) x (N x 24 x 16)
            T = tf.reshape(
                tf.matmul(W, tf.reshape(A, [num_batch, 24, 16])),
                [num_batch, -1, 4, 4])
            v_posed_homo = tf.concat(
                [v_posed, tf.ones([num_batch, v_posed.shape[1], 1])], 2)
            v_homo = tf.matmul(T, tf.expand_dims(v_posed_homo, -1))

            verts = v_homo[:, :, :3, 0]
            verts_t = verts + tf.expand_dims(trans, axis=1)

            return verts_t, v_shaped_personal, v_shaped
