import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Input, Flatten, Dense, Lambda, Conv2D, MaxPool2D, Average, Concatenate, Add, Reshape
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tqdm.keras import TqdmCallback

from smpl.smpl_layer import SmplTPoseLayer, SmplBody25FaceLayer
from graphconv.graphconvlayer import GraphConvolution
from render.render_layer import RenderLayer

from smpl.batch_lbs import batch_rodrigues
from smpl.bodyparts import regularize_laplace, regularize_symmetry
from lib.geometry import compute_laplacian_diff, sparse_to_tensor
from graphconv.util import sparse_dot_adj_batch, chebyshev_polynomials
from render.render import perspective_projection

import pickle as pkl
import dill
import imageio
import trimesh
dill._dill._reverse_typemap["ObjectType"] = object

def NameLayer(name):
    return Lambda(lambda i: i, name=name)

def laplace_mse(_, ypred):
    w = regularize_laplace()
    return K.mean(w[np.newaxis, :, np.newaxis] * K.square(ypred), axis=-1)

def symmetry_mse(_, ypred):
    w = regularize_symmetry()

    idx = np.load(os.path.join(os.path.dirname(__file__), '../assets/vert_sym_idxs.npy'))
    ypred_mirror = tf.gather(ypred, idx, axis=1) * np.array([-1., 1., 1.]).astype(np.float32).reshape(1, 1, 3)

    return K.mean(w[np.newaxis, :, np.newaxis] * K.square(ypred - ypred_mirror), axis=-1)

def reprojection(fl, cc, w, h):
    def _r(ytrue, ypred):
        b_size = tf.shape(input=ypred)[0]
        projection_matrix = perspective_projection(fl, cc, w, h, .1, 10)
        projection_matrix = tf.tile(tf.expand_dims(projection_matrix, 0), (b_size, 1, 1))

        ypred_h = tf.concat([ypred, tf.ones_like(ypred[:, :, -1:])], axis=2)
        ypred_proj = tf.matmul(ypred_h, projection_matrix)
        ypred_proj /= tf.expand_dims(ypred_proj[:, :, -1], -1)
        return K.mean(K.square((ytrue[:, :, :2] - ypred_proj[:, :, :2]) * tf.expand_dims(ytrue[:, :, 2], -1)))
    return _r

class Octopus(object):
    def __init__(self, num=8, img_size=1080):
        self.num = num
        self.img_size = img_size
        self.inputs = []
        self.poses = []
        self.ts = []

        images = [Input(shape=(self.img_size, self.img_size, 3), name='image_{}'.format(i)) for i in range(self.num)]
        Js = [Input(shape=(25, 3), name='J_2d_{}'.format(i)) for i in range(self.num)]
        # rgbs = [Input(shape=(self.img_size, self.img_size, 3), name='rgb_{}'.format(i)) for i in range(self.num)]

        self.inputs.extend(images)
        self.inputs.extend(Js)

        self.uv = tf.Variable(np.array(imageio.imread(
            os.path.join(os.path.dirname(__file__), '../assets/smpl_part/smpl_uv.png')), 'float32')[..., :-1] / 255.,
                              trainable=True, name='UV')
        # self.uv = tf.Variable(np.ones((1024, 1024, 3)), trainable=True)
        smpl_mesh = trimesh.load(os.path.join(os.path.dirname(__file__), '../assets/smpl_part/smpl_uv.obj'))
        texture_dict = dict(np.load(os.path.join(os.path.dirname(__file__), '../assets/smpl_part/smpl_texture_dict.npy'), allow_pickle=True).item())
        smpl_mesh_uv = np.array(smpl_mesh.visual.uv, 'float32') * self.uv.shape[0]

        smpl_mesh_u = smpl_mesh_uv[:, 0]
        smpl_mesh_u_c = np.ceil(smpl_mesh_uv[:, 0])
        smpl_mesh_u_f = np.floor(smpl_mesh_uv[:, 0])
        smpl_mesh_v = smpl_mesh_uv[:, 1]
        smpl_mesh_v_c = np.ceil(smpl_mesh_uv[:, 1])
        smpl_mesh_v_f = np.floor(smpl_mesh_uv[:, 1])

        uv_ff = tf.gather_nd(self.uv, indices=tf.constant(np.stack((smpl_mesh_v_f, smpl_mesh_u_f), axis=1), dtype=tf.int64))
        uv_cf = tf.gather_nd(self.uv, indices=tf.constant(np.stack((smpl_mesh_v_f, smpl_mesh_u_c), axis=1), dtype=tf.int64))
        uv_cc = tf.gather_nd(self.uv, indices=tf.constant(np.stack((smpl_mesh_v_c, smpl_mesh_u_c), axis=1), dtype=tf.int64))
        uv_fc = tf.gather_nd(self.uv, indices=tf.constant(np.stack((smpl_mesh_v_c, smpl_mesh_u_f), axis=1), dtype=tf.int64))

        uc = (smpl_mesh_u_c - smpl_mesh_u)[..., None]
        uf = (smpl_mesh_u - smpl_mesh_u_f)[..., None]
        vc = (smpl_mesh_v_c - smpl_mesh_v)[..., None]
        vf = (smpl_mesh_v - smpl_mesh_v_f)[..., None]
        smpl_color = vc * (uc * uv_ff + uf * uv_cf) + vf * (uc * uv_fc + uf * uv_cc)
        vertex_colors = []
        for i in range(6890):
            colors = tf.reduce_mean(tf.gather(smpl_color, texture_dict[i]), axis=0, keepdims=True)
            vertex_colors.append(colors)
        vertex_colors = tf.concat(vertex_colors, axis=0)

        pose_raw = np.load(os.path.join(os.path.dirname(__file__), '../assets/mean_a_pose.npy'))
        pose_raw[:3] = 0.
        pose = tf.reshape(batch_rodrigues(pose_raw.reshape(-1, 3).astype(np.float32)), (-1, ))
        trans = np.array([0., 0.2, -2.3])

        batch_size = tf.shape(input=images[0])[0]

        conv2d_0 = Conv2D(8, (3, 3), strides=(2, 2), activation='relu', kernel_initializer='he_normal', name='conv2d_1', trainable=False)
        maxpool_0 = MaxPool2D((2, 2), name='max_pooling2d_1')

        conv2d_1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', name='conv2d_2', trainable=False)
        maxpool_1 = MaxPool2D((2, 2), name='max_pooling2d_2')

        conv2d_2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', name='conv2d_3', trainable=False)
        maxpool_2 = MaxPool2D((2, 2), name='max_pooling2d_3')

        conv2d_3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', name='conv2d_4', trainable=False)
        maxpool_3 = MaxPool2D((2, 2), name='max_pooling2d_4')

        conv2d_4 = Conv2D(128, (3, 3), name='conv2d_5', trainable=False)
        maxpool_4 = MaxPool2D((2, 2), name='max_pooling2d_5')

        flat = Flatten()
        self.image_features = flat

        latent_code = Dense(20, name='latent_shape')

        self.pose_trans_tmp = tf.expand_dims(tf.concat((trans, pose), axis=0),0)
        posetrans_init = tf.tile(self.pose_trans_tmp, (batch_size, 1))

        J_flat = Flatten()
        concat_pose = Concatenate()

        latent_pose_from_I = Dense(200, name='latent_pose_from_I', activation='relu', trainable=False)
        latent_pose_from_J = Dense(200, name='latent_pose_from_J', activation='relu', trainable=False)
        latent_pose = Dense(100, name='latent_pose')
        posetrans_res = Dense(24 * 3 * 3 + 3, name='posetrans_res',
                              kernel_initializer=RandomNormal(stddev=0.01), trainable=False)
        posetrans = Add(name='posetrans')

        dense_layers = []

        for i, (J, image) in enumerate(zip(Js, images)):
            conv2d_0_i = conv2d_0(image)
            maxpool_0_i = maxpool_0(conv2d_0_i)

            conv2d_1_i = conv2d_1(maxpool_0_i)
            maxpool_1_i = maxpool_1(conv2d_1_i)

            conv2d_2_i = conv2d_2(maxpool_1_i)
            maxpool_2_i = maxpool_2(conv2d_2_i)

            conv2d_3_i = conv2d_3(maxpool_2_i)
            maxpool_3_i = maxpool_3(conv2d_3_i)

            conv2d_4_i = conv2d_4(maxpool_3_i)
            maxpool_4_i = maxpool_4(conv2d_4_i)

            # shape
            flat_i = flat(maxpool_4_i)

            latent_code_i = latent_code(flat_i)
            dense_layers.append(latent_code_i)

            # pose
            J_flat_i = J_flat(J)
            latent_pose_from_I_i = latent_pose_from_I(flat_i)
            latent_pose_from_J_i = latent_pose_from_J(J_flat_i)

            concat_pose_i = concat_pose([latent_pose_from_I_i, latent_pose_from_J_i])
            latent_pose_i = latent_pose(concat_pose_i)
            posetrans_res_i = posetrans_res(latent_pose_i)
            posetrans_i = posetrans([posetrans_res_i, posetrans_init])

            self.poses.append(
                Lambda(lambda x: tf.reshape(x[:, 3:], (-1, 24, 3, 3)), name='pose_{}'.format(i))(posetrans_i)
            )
            self.ts.append(
                Lambda(lambda x: x[:, :3], name='trans_{}'.format(i))(posetrans_i)
            )

        if self.num > 1:
            self.dense_merged = Average(name='merged_latent_shape')(dense_layers)
        else:
            self.dense_merged = NameLayer(name='merged_latent_shape')(dense_layers[0])

        # betas
        self.betas = Dense(10, name='betas', trainable=False)(self.dense_merged)

        with open(os.path.join(os.path.dirname(__file__), '../assets/smpl_sampling.pkl'), 'rb') as f:
            sampling = pkl.load(f, encoding='iso-8859-1')

        M = sampling['meshes']
        U = sampling['up']
        D = sampling['down']
        A = sampling['adjacency']

        self.faces = M[0]['f'].astype(np.int32)

        low_res = D[-1].shape[0]
        tf_U = [sparse_to_tensor(u) for u in U]
        tf_A = [map(sparse_to_tensor, chebyshev_polynomials(a, 3)) for a in A]

        shape_features_dense = Dense(low_res * 64, kernel_initializer=RandomNormal(stddev=0.003),
                                     name='shape_features_flat')(self.dense_merged)
        shape_features = Reshape((low_res, 64), name="shape_features")(shape_features_dense)

        conv_l3 = GraphConvolution(32, tf_A[3], activation='relu', name='conv_l3', trainable=False)(shape_features)
        unpool_l2 = Lambda(lambda v: sparse_dot_adj_batch(tf_U[2], v), name='unpool_l2')(conv_l3)
        conv_l2 = GraphConvolution(16, tf_A[2], activation='relu', name='conv_l2', trainable=False)(unpool_l2)
        unpool_l1 = Lambda(lambda v: sparse_dot_adj_batch(tf_U[1], v), name='unpool_l1')(conv_l2)
        conv_l1 = GraphConvolution(16, tf_A[1], activation='relu', name='conv_l1', trainable=False)(unpool_l1)
        unpool_l0 = Lambda(lambda v: sparse_dot_adj_batch(tf_U[0], v), name='unpool_l0')(conv_l1)
        conv_l0 = GraphConvolution(3, tf_A[0], activation='tanh', name='offsets_pre')(unpool_l0)

        self.offsets = Lambda(lambda x: x / 10., name='offsets')(conv_l0)

        smpl = SmplTPoseLayer(theta_in_rodrigues=False, theta_is_perfect_rotmtx=False, name='smpl_t_pose_layer_1')
        smpls = [NameLayer('smpl_{}'.format(i))(smpl([p, self.betas, t, self.offsets])) for i, (p, t) in
                 enumerate(zip(self.poses, self.ts))]

        self.vertices = [Lambda(lambda s: s[0], name='vertices_{}'.format(i))(smpl) for i, smpl in enumerate(smpls)]

        # we only need one instance per batch for laplace
        self.vertices_tposed = Lambda(lambda s: s[1], name='vertices_tposed')(smpls[0])
        vertices_naked = Lambda(lambda s: s[2], name='vertices_naked')(smpls[0])

        def laplacian_function(x):
            faces = self.faces
            v0, v1 = x
            return compute_laplacian_diff(v0, v1, faces)
        self.laplacian = Lambda(lambda v: laplacian_function(v), name='laplacian')([self.vertices_tposed, vertices_naked])
        self.symmetry = NameLayer('symmetry')(self.vertices_tposed)

        smplf = SmplBody25FaceLayer(theta_in_rodrigues=False, theta_is_perfect_rotmtx=False)
        kps = [NameLayer('kps_{}'.format(i))(smplf([p, self.betas, t]))
               for i, (p, t) in enumerate(zip(self.poses, self.ts))]

        self.Js = [Lambda(lambda jj: jj[:, :25], name='J_reproj_{}'.format(i))(j) for i, j in enumerate(kps)]
        self.face_kps = [Lambda(lambda jj: jj[:, 25:], name='face_reproj_{}'.format(i))(j) for i, j in enumerate(kps)]

        self.repr_loss = reprojection([self.img_size, self.img_size],
                                      [self.img_size / 2., self.img_size / 2.],
                                      self.img_size, self.img_size)

        renderer = RenderLayer(self.img_size, self.img_size, 1, np.ones((6890, 1)), np.zeros(1), self.faces,
                               [self.img_size, self.img_size], [self.img_size / 2., self.img_size / 2.],
                               name='render_layer')
        self.rendered = [NameLayer('rendered_{}'.format(i))(renderer(v)) for i, v in enumerate(self.vertices)]

        # color
        renderer_color = RenderLayer(self.img_size, self.img_size, 3, vertex_colors, np.ones(3), self.faces,
                                     [self.img_size, self.img_size], [self.img_size / 2., self.img_size / 2.],
                                     name='render_color_layer')
        self.rendered_color = [NameLayer('rendered_color_{}'.format(i))(renderer_color(v)) for i, v in enumerate(self.vertices)]

        # texture loss
        # texture_loss = NameLayer('')
        # self.texture_loss = NameLayer('rendered_color')(sum(tf.keras.losses.mse(y, p) for y, p in zip(rgbs, self.rendered_color)))

        self.inference_model = Model(
            inputs=self.inputs,
            outputs=[self.vertices_tposed] + self.vertices + [self.betas, self.offsets] + self.poses + self.ts + self.rendered_color
        )

        self.opt_pose_model = Model(
            inputs=self.inputs,
            outputs=self.Js
        )

        opt_pose_loss = {'J_reproj_{}'.format(i): self.repr_loss for i in range(self.num)}
        self.opt_pose_model.compile(loss=opt_pose_loss, optimizer='adam')

        self.opt_shape_model = Model(
            inputs=self.inputs,
            outputs=self.Js + self.face_kps + self.rendered + [self.symmetry, self.laplacian]
        )

        self.opt_texture_model = Model(
            inputs=self.inputs,
            outputs=self.rendered_color
        )

        opt_shape_loss = {
            'laplacian': laplace_mse,
            'symmetry': symmetry_mse,
        }
        opt_shape_weights = {
            'laplacian': 100. * self.num,
            'symmetry': 50. * self.num,
        }
        opt_texture_loss = {
            # 'rendered_color': lambda _, __: self.texture_loss
        }
        # opt_texture_weights = {}
        for i in range(self.num):
            opt_shape_loss['rendered_{}'.format(i)] = 'mse'
            opt_shape_weights['rendered_{}'.format(i)] = 1.

            opt_shape_loss['J_reproj_{}'.format(i)] = self.repr_loss
            opt_shape_weights['J_reproj_{}'.format(i)] = 50.

            opt_shape_loss['face_reproj_{}'.format(i)] = self.repr_loss
            opt_shape_weights['face_reproj_{}'.format(i)] = 10. * self.num

            opt_texture_loss['rendered_color_{}'.format(i)] = 'mse'
            # opt_texture_weights['rendered_color_{}'.format(i)] = 1.

        self.opt_shape_model.compile(loss=opt_shape_loss, loss_weights=opt_shape_weights, optimizer='adam')
        adam_opt = tf.keras.optimizers.Adam(learning_rate=.01)
        # self.adam_opt = adam_opt.minimize(self.texture_loss, var_list=[self.uv])
        self.opt_texture_model.compile(loss=opt_texture_loss, optimizer=adam_opt)

    def load(self, checkpoint_path):
        self.inference_model.load_weights(checkpoint_path, by_name=True)

    def opt_texture(self, segmentations, joints_2d, images, opt_steps):
        data = {}
        supervision = {}

        for i in range(self.num):
            data['image_{}'.format(i)] = np.tile(
                np.float32(segmentations[i].reshape((1, self.img_size, self.img_size, -1))),
                (opt_steps, 1, 1, 1)
            )
            data['J_2d_{}'.format(i)] = np.tile(
                np.float32(np.expand_dims(joints_2d[i], 0)),
                (opt_steps, 1, 1)
            )

            supervision['rendered_color_{}'.format(i)] = np.tile(
                np.expand_dims(images[i], axis=0), (opt_steps, 1, 1, 1)
            )

        self.opt_texture_model.fit(
            data, supervision,
            batch_size=1, epochs=1, verbose=0,
            callbacks=[TqdmCallback(verbose=1)]
        )

    def opt_pose(self, segmentations, joints_2d, opt_steps):
        data = {}
        supervision = {}

        for i in range(self.num):
            data['image_{}'.format(i)] = np.tile(
                np.float32(segmentations[i].reshape((1, self.img_size, self.img_size, -1))),
                (opt_steps, 1, 1, 1)
            )
            data['J_2d_{}'.format(i)] = np.tile(
                np.float32(np.expand_dims(joints_2d[i], 0)),
                (opt_steps, 1, 1)
            )
            supervision['J_reproj_{}'.format(i)] = np.tile(
                np.float32(np.expand_dims(joints_2d[i], 0)),
                (opt_steps, 1, 1)
            )

        self.opt_pose_model.fit(
            data, supervision,
            batch_size=1, epochs=1, verbose=0,
            callbacks = [TqdmCallback(verbose=1)]
        )

    def opt_shape(self, segmentations, joints_2d, face_kps, opt_steps):
        data = {}
        supervision = {
            'laplacian': np.zeros((opt_steps, 6890, 3)),
            'symmetry': np.zeros((opt_steps, 6890, 3)),
        }

        for i in range(self.num):
            data['image_{}'.format(i)] = np.tile(
                np.float32(segmentations[i].reshape((1, self.img_size, self.img_size, -1))),
                (opt_steps, 1, 1, 1)
            )
            data['J_2d_{}'.format(i)] = np.tile(
                np.float32(np.expand_dims(joints_2d[i], 0)),
                (opt_steps, 1, 1)
            )

            supervision['J_reproj_{}'.format(i)] = np.tile(
                np.float32(np.expand_dims(joints_2d[i], 0)),
                (opt_steps, 1, 1)
            )
            supervision['face_reproj_{}'.format(i)] = np.tile(
                np.float32(np.expand_dims(face_kps[i], 0)),
                (opt_steps, 1, 1)
            )
            supervision['rendered_{}'.format(i)] = np.tile(
                np.expand_dims(
                    np.any(np.float32(segmentations[i].reshape((1, self.img_size, self.img_size, -1)) > 0), axis=-1),
                    -1),
                (opt_steps, 1, 1, 1)
            )
        self.opt_shape_model.fit(
            data, supervision,
            batch_size=1, epochs=1, verbose=0, callbacks=[TqdmCallback(verbose=1)]
        )

    def predict(self, segmentations, joints_2d):
        data = {}

        for i in range(self.num):
            data['image_{}'.format(i)] = np.float32(segmentations[i].reshape((1, self.img_size, self.img_size, -1)))
            data['J_2d_{}'.format(i)] = np.float32(np.expand_dims(joints_2d[i], 0))

        pred = self.inference_model.predict(data)

        res = {
            'vertices_tposed': pred[0][0],
            'vertices': np.array([p[0] for p in pred[1:self.num + 1]]),
            'faces': self.faces,
            'betas': pred[self.num + 1][0],
            'offsets': pred[self.num + 2][0],
            'poses': np.array(
                [cv2.Rodrigues(p0)[0] for p in pred[self.num + 3:2 * self.num + 3] for p0 in p[0]]
            ).reshape((self.num, -1)),
            'trans': np.array([t[0] for t in pred[2 * self.num + 3:]]),
            'rendered_color': np.array(pred[-self.num:])
        }

        return res


if __name__ == "__main__":
    octopus = Octopus()
    octopus.inference_model.summary()
