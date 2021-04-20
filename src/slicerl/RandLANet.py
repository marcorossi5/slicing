import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Activation,
    Reshape,
    Concatenate,
    Dropout,
    Conv1D
)
import knn

TF_DTYPE_INT = tf.int32
TF_DTYPE     = tf.float32

NP_DTYPE_INT = np.int32
NP_DTYPE     = np.float32

def float_me(x):
    return tf.constant(x, dtype=TF_DTYPE)

def int_me(x):
    return tf.constant(x, dtype=TF_DTYPE_INT)

#======================================================================
class LocSE(Layer):
    """ Class defining Local Spatial Encoding Layer. """
    #----------------------------------------------------------------------
    def __init__(self, K=8, activation='relu', **kwargs):
        """
        Parameters
        ----------
            - K          : int, number of nearest neighbors
            - activation : str, MLP layer activation
        """
        super(LocSE, self).__init__(**kwargs)
        self.K          = K
        self.activation = activation
        self._cache     = None
        self._is_cached = False

    #----------------------------------------------------------------------
    @property
    def cache(self):
        return self._cache

    #----------------------------------------------------------------------
    @cache.setter
    def cache(self, cache):
        """
        Parameters
        ----------
            - cache : list, [n_points, n_feats] knn cache
        """
        self._cache     = cache
        self._is_cached = True

    #----------------------------------------------------------------------
    def build(self, input_shape):
        """
        Note: the MLP kernel has size 1. This means that point features are
              not mixed across neighbours.

        Parameters
        ----------
            input_shape : list of tf.TensorShape, first is (B,N,spatial dims),
                          second is (B,N,feature number)

        """
        spatial, features = input_shape
        B , N , dims   = spatial
        B_, N_, n_feat = features
        assert B == B_, f"Batches number mismatch in input point cloud, got {B} and {B_}"
        assert N == N_, f"Points number mismatch in input point cloud, got {N} and {N_}"

        self.MLP = Conv1D(n_feat, 1, input_shape=(N, dims),
                          activation=self.activation, 
                          kernel_initializer='glorot_uniform',
                          bias_initializer='zeros',
                          name='MLP')
        self.cat = Concatenate(axis=-1 , name='cat')

    #----------------------------------------------------------------------
    def call(self, inputs, cached=False):
        """
        Layer forward pass.

        Parameters
        ----------
            - inputs     : list of tf.Tensors, tensors describing spatial and
                           feature point cloud of shapes=[(B,N,dims), (B,N,f_dims)]
            - no_knn     : bool, wether to use cached relative position

        Returns
        -------
            tf.Tensor of shape=(B,N,K,2*f_dims)
        
        Note: dims (spatial dimensions) equal to 2
        """
        pc, feats = inputs
        if cached and self._is_cached:
            rppe, n_feats = self._cache
        else:
            n_idx    = knn.knn_batch(pc, pc, K=self.K, omp=True)
            n_points = self.gather_neighbours(pc, n_idx)
            n_feats  = self.gather_neighbours(feats, n_idx)
            
            # point cloud with K expanded axis
            Kpc   = tf.repeat(tf.expand_dims(pc,-2), self.K, -2)
            relp  = Kpc - n_points
            norms = tf.norm(relp, ord='euclidean', axis=-1, keepdims=True, name='norm')

            # relative point position encoding
            rppe = self.cat([Kpc, n_points, relp, norms])
            self._cache     = [rppe, n_feats]
            self._is_cached = True
        
        r = self.MLP(rppe)
        return self.cat([n_feats, r])

    #----------------------------------------------------------------------
    def get_config(self):
        config = super(Linear, self).get_config()
        config.update({
            "units"      : self.units,
            "K"          : self.K,
            "activation" : self.activation
        })
        return config

    #----------------------------------------------------------------------
    @staticmethod
    def gather_neighbours(pc, n_idx):
        """
        Parameters
        ----------
            - pc    : tf.Tensor, point cloud tensor of shape=(B,N,dims)
            - n_idx : tf.Tensor, neighbourhood index tensor of shape=(B,N,K)
        
        Returns
        -------
            tf.Tensor of shape=(B,N,K,dims)
        """
        B, N, dims = tf.shape(pc, name='pc_shape')
        K          = tf.shape(n_idx, name='K_shape')[-1]
        idx_input  = tf.reshape(n_idx, [B,-1])
        features   = tf.gather(pc, idx_input, axis=1, batch_dims=1, name='gather_neighbours')
        return tf.reshape(features, [B,N,K,dims])

#======================================================================
class AttentivePooling(Layer):
    """ Class defining Attentive Pooling Layer. """
    #----------------------------------------------------------------------
    def __init__(self, units=1, activation='relu', **kwargs):
        """
        Parameters
        ----------
            - units      : int, number of output units
            - activation : str, MLP layer activation
        """
        super(AttentivePooling, self).__init__(**kwargs)
        self.units      = units
        self.activation = activation

    #----------------------------------------------------------------------
    def build(self, input_shape):
        """
        Note: the MLP kernel has size 1. This means that point features are
              not mixed across neighbours.
        """

        self.MLP_score = Conv1D(input_shape[-1], 1, input_shape=input_shape[-2:],
                          activation='softmax', 
                          kernel_initializer='glorot_uniform',
                          bias_initializer='zeros',
                          name='attention_score_MLP')
        self.MLP_final = Conv1D(self.units, 1, input_shape=input_shape[-2:],
                          activation=self.activation, 
                          kernel_initializer='glorot_uniform',
                          bias_initializer='zeros',
                          name='final_MLP')
        self.reshape   = Reshape((-1, self.units), name='reshape')

    #----------------------------------------------------------------------
    def call(self, n_feats):
        """
        Layer forward pass.

        Parameters
        ----------
            - n_feats : tf.Tensor, point cloud tensor of shape=(B,N,dims)
        
        Returns
        -------
            tf.Tensor of shape=(B,N,units)
        """
        scores = self.MLP_score(n_feats)
        attention = tf.math.reduce_sum(n_feats*scores, axis=-2, keepdims=True)

        return self.reshape( self.MLP_final(attention) )

    #----------------------------------------------------------------------
    def get_config(self):
        config = super(AttentivePooling, self).get_config()
        config.update({
            "units"      : self.units,
            "activation" : self.activation
        })
        return config

#======================================================================
class DilatedResBlock(Layer):
    """ Class defining Dilated Residual Block. """
    #----------------------------------------------------------------------
    def __init__(self, units=1, K=8, activation='relu', **kwargs):
        """
        Parameters
        ----------
            - units      : int, number of output units, divisible by 4
            - K          : int, number of nearest neighbors
            - activation : str, MLP layer activation
        """
        super(DilatedResBlock, self).__init__(**kwargs)
        self.units      = units
        self.K          = K
        self.activation = activation

    #----------------------------------------------------------------------
    def build(self, input_shape):
        """
        Note: the MLP kernel has size 1. This means that point features are
              not mixed across neighbours.
        """
        spatial, features = input_shape
        B , N , dims   = spatial
        B_, N_, n_feat = features
        assert B == B_, f"Batches number mismatch in input point cloud, got {B} and {B_}"
        assert N == N_, f"Points number mismatch in input point cloud, got {N} and {N_}"

        self.MLP_0   = Conv1D(self.units//4, 1, input_shape=(N, n_feat),
                              activation=self.activation, 
                              kernel_initializer='glorot_uniform',
                              bias_initializer='zeros',
                              name='MLP_0')
        self.MLP_1   = Conv1D(self.units, 1, input_shape=(N, self.units//2),
                              activation=self.activation, 
                              kernel_initializer='glorot_uniform',
                              bias_initializer='zeros',
                              name='MLP_2')
        self.MLP_res = Conv1D(self.units, 1, input_shape=(N, n_feat),
                              activation=self.activation, 
                              kernel_initializer='glorot_uniform',
                              bias_initializer='zeros',
                              name='MLP_res')

        self.locse_0 = LocSE(K=self.K, name='locse_0')
        self.locse_1 = LocSE(K=self.K, name='locse_1')

        self.att_0 = AttentivePooling(self.units//4, name='attention_0')
        self.att_1 = AttentivePooling(self.units//2, name='attention_1')

        self.act = Activation(tf.nn.leaky_relu, name='lrelu')

    #----------------------------------------------------------------------
    def call(self, inputs):
        """
        Layer forward pass.

        Parameters
        ----------
            - inputs : list of tf.Tensors, tensors describing spatial and
                       feature point cloud of shapes=[(B,N,dims), (B,N,f_dims)]
        
        Returns
        -------
            tf.Tensor of shape=(B,N,units)
        """
        pc, feats = inputs

        # residual connection
        y = self.MLP_res(feats)

        # dilation block
        x = self.MLP_0(feats)
        x = self.att_0( self.locse_0([pc, x]) )
        
        # store cash in locse_1 to skip knn building
        self.locse_1.cache = self.locse_0.cache

        x = self.att_1( self.locse_1([pc, x], cached=True) )
        x = self.MLP_1(x)

        return self.act(x + y)

    #----------------------------------------------------------------------
    def get_config(self):
        config = super(DilatedResBlock, self).get_config()
        config.update({
            "units"      : self.units,
            "K"          : self.K,
            "activation" : self.activation
        })
        return config


#======================================================================
class RandomSample(Layer):
    """ Class defining Random Sampling Layer. """
    #----------------------------------------------------------------------
    def __init__(self, scale=2, **kwargs):
        """
        Parameters
        ----------
            - scale: int, scaling factor of the input cloud
        """
        super(RandomSample, self).__init__(**kwargs)
        self.scale = scale

    #----------------------------------------------------------------------
    def call(self, inputs):
        """
        Parameters
        ----------
            - inputs : list of tf.Tensors, tensors describing spatial and
                       feature point cloud of shapes=[(B,N,dims), (B,N,f_dims)]
        
        Returns
        -------
            - tf.Tensor, point cloud of shape=(B,N//scale,dims)
            - tf.Tensor, features of shape=(B,N//scale,f_dims)
        """
        pc, feats = inputs

        B, N, dims = tf.shape(pc)
        B_, N_, f_dims = tf.shape(feats)
        assert B == B_, f"Batches number mismatch in input point cloud, got {B} and {B_}"
        assert N == N_, f"Points number mismatch in input point cloud, got {N} and {N_}"

        rnds = [tf.random.shuffle(tf.range(N)) for i in range(B)]
        rnds = tf.stack(rnds)

        valid_idx   = rnds[:,:N//2]
        invalid_idx = rnds[:,N//2:]

        # downsample point cloud
        valid_pc = tf.gather(pc, valid_idx, axis=1, batch_dims=1, name='downsample_pc')
        valid_feats = tf.gather(feats, valid_idx, axis=1, batch_dims=1, name='downsample_features')

        # store info for upsampling
        invalid_pc = tf.gather(pc, invalid_idx, axis=1, batch_dims=1)
        n_idx = int_me(knn.knn_batch(valid_pc, invalid_pc, K=2, omp=True)[:,:,1])
        self.upsample_idx = [n_idx, rnds]

        return valid_pc, valid_feats

    #----------------------------------------------------------------------
    def get_config(self):
        config = super(RandomSample, self).get_config()
        config.update({
            "scale"          : self.scale,
        })
        return config

#======================================================================
class UpSample(Layer):
    """ Class defining Upsampling Layer. """
    #----------------------------------------------------------------------
    def __init__(self, scale=2, **kwargs):
        """
        Parameters
        ----------
            - scale: int, scaling factor of the input cloud
        """
        super(UpSample, self).__init__(**kwargs)
        self.scale = scale
        self.cat = Concatenate(axis=1, name='cat')

    #----------------------------------------------------------------------
    def call(self, inputs):
        """
        Parameters
        ----------
            - inputs : list of tf.Tensors,
                       feature point cloud of shape=(B,N//scale,f_dims),
                       interpolating indices of shape=(B,N//scale),
                       upsampling indices of shape=(B,N)
        
        Returns
        -------
            - tf.Tensor, features of shape=(B,N,f_dims)
        """
        feats, interpolate_idx, upsample_idx = inputs

        copy_feats = tf.gather(feats, interpolate_idx, axis=1,
                                       batch_dims=1, name='copy_features')

        interpolated_feats = self.cat([feats, copy_feats])

        shape = tf.shape(interpolated_feats)
        
        first = tf.repeat(tf.expand_dims(tf.range(shape[0]), 1), shape[1], axis=1)
        indices = tf.stack([first, upsample_idx], axis=-1)

        return tf.scatter_nd(indices, interpolated_feats, shape, name='upsample_features')

    #----------------------------------------------------------------------
    def get_config(self):
        config = super(UpSample, self).get_config()
        config.update({
            "scale"          : self.scale,
        })
        return config

#======================================================================
class RandLANet(Model):
    """ Class deifining RandLA-Net. """
    def __init__(self, input_f_dims=2, K=16, scale_factor=2, nb_layers=4, **kwargs):
        """
        Parameters
        ----------
            - K            : int, number of nearest neighbours to find
            - scale_factor : int, scale factor for down/up-sampling
            - nb_layers    : int, number of inner enocding/decoding layers
        """
        super(RandLANet, self).__init__(**kwargs)

        # store args
        self.input_f_dims = input_f_dims
        self.K = K
        self.scale_factor = scale_factor
        self.nb_layers = nb_layers

        # store some useful parameters
        self.num_classes = 2
        self.fc_units = [8, 64, 32, self.num_classes]
        self.fc_acts  = ['relu']*3 + ['softmax']
        self.latent_f_dim = 32
        self.enc_units = [
            self.latent_f_dim*self.scale_factor**i \
                for i in range(self.nb_layers)
                           ]
        self.dec_units = self.enc_units[-2::-1] + [self.fc_units[0]]

        # build layers
        # self.input = Input(shape=(None, input_f_dims) , name='input') # TODO: check if this is needed
        self.fcs = [
            Dense(units, activation=act, name=f'fc{i}') \
                for i, (units, act) in enumerate(zip(self.fc_units, self.fc_acts))
        ]

        self.encoding   = self.build_encoder()
        self.middle_MLP = Conv1D(self.enc_units[-1], 1,
                              activation='relu', 
                              kernel_initializer='glorot_uniform',
                              bias_initializer='zeros',
                              name='MLP_middle')
        self.decoding   = self.build_decoder()

    #----------------------------------------------------------------------
    def build_encoder(self):
        self.encoder = [
            DilatedResBlock(units=units, K=self.K, name=f'encoder_DRB{i}') \
                                for i, units in enumerate(self.enc_units)
                       ]
        self.RSs = [
            RandomSample(self.scale_factor, name=f'encoder_RS{i}') \
                for i in range(self.nb_layers)
                   ]

    #----------------------------------------------------------------------
    def build_decoder(self):
        self.decoder = [
            Conv1D(units, 1, activation='relu', kernel_initializer='glorot_uniform',
                   bias_initializer='zeros', name=f'upsample_MLP{i}') \
                for i, units in enumerate(self.dec_units)
                       ]
        self.cat = Concatenate(axis=-1, name='upsample_cat')
        self.USs = [
            UpSample(self.scale_factor, name=f'upsample_US{i}') \
                for i in range(self.nb_layers)
                   ]

    #----------------------------------------------------------------------
    def call(self, inputs):
        """
        Parameters
        ----------
            - inputs : list of tf.Tensors, point cloud of shape=(B,N,dims) and
                       feature point cloud of shape=(B,N,f_dims)

        Returns
        -------
            tf.Tensor
        """
        pc, feats = inputs

        residuals = []
        rs_idxs   = []

        feats = self.fcs[0](feats)
        residuals.append(feats)
        for drb, rs  in zip(self.encoder, self.RSs):
            feats = drb( [pc, feats] )
            pc, feats = rs([pc, feats])
            rs_idxs.append(rs.upsample_idx)
            residuals.append(feats)
        
        feats = self.middle_MLP(feats)

        for us, mlp, rs_idx, res in zip(self.USs, self.decoder, rs_idxs[::-1], residuals[-2::-1]):
            feats = us([feats, *rs_idx])
            feats = mlp( self.cat([res,feats]) )

        for fc in self.fcs[1:]:
            feats = fc(feats)
        return feats
        # TODO: check model step by step
