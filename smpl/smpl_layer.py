import tensorflow as tf
from smpl.batch_smpl import SMPL
from smpl.joints import joints_body25, face_landmarks
#from keras.engine.topology import Layer
from tensorflow.keras.layers import Layer

# From joints.py
import os
import sys
import pickle as pkl
from lib.geometry import sparse_to_tensor, sparse_dense_matmul_batch_tile



class SmplTPoseLayer(Layer):

    def __init__(self, model='assets/neutral_smpl.pkl', theta_in_rodrigues=False, theta_is_perfect_rotmtx=True, **kwargs):
        self.smpl = SMPL(model, theta_in_rodrigues, theta_is_perfect_rotmtx)
        super(SmplTPoseLayer, self).__init__(**kwargs)

    #def call(self, (pose, betas, trans, v_personal)):
    def call(self, inputs):
        pose, betas, trans, v_personal = inputs
        verts, v_shaped_personal, v_shaped = self.smpl([pose, betas, trans, v_personal])

        # return [verts, self.smpl.v_shaped_personal, self.smpl.v_shaped]
        return [verts, v_shaped_personal, v_shaped]

    def compute_output_shape(self, input_shape):
        shape = input_shape[0][0], 6890, 3

        return [shape, shape, shape]


class SmplBody25FaceLayer(Layer):

    def __init__(self, model='assets/neutral_smpl.pkl', theta_in_rodrigues=False, theta_is_perfect_rotmtx=True, **kwargs):
        self.smpl = SMPL(model, theta_in_rodrigues, theta_is_perfect_rotmtx)
        self.body_25_reg = sparse_to_tensor(
            #pkl.load(open(os.path.join(os.path.dirname(__file__), '../assets/J_regressor.pkl'), 'rb')).T
            pkl.load(open(os.path.join(os.path.dirname(__file__), '../assets/J_regressor.pkl'), 'rb'), encoding='iso-8859-1').T
        )
        self.face_reg = sparse_to_tensor(
            # pkl.load(open(os.path.join(os.path.dirname(__file__), '../assets/face_regressor.pkl'), 'rb')).T
            pkl.load(open(os.path.join(os.path.dirname(__file__), '../assets/face_regressor.pkl'), 'rb'),
                     encoding='iso-8859-1').T
        )
        super(SmplBody25FaceLayer, self).__init__(**kwargs)

    #def call(self, (pose, betas, trans)):
    def call(self, inputs):
        pose, betas, trans = inputs
        v_personal = tf.tile(tf.zeros((1, 6890, 3)), (tf.shape(input=betas)[0], 1, 1))

        v, _, _ = self.smpl([pose, betas, trans, v_personal])
        jkps = sparse_dense_matmul_batch_tile(self.body_25_reg, v)
        fkps = sparse_dense_matmul_batch_tile(self.face_reg, v)
        return tf.concat((jkps, fkps), axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], 95, 3

