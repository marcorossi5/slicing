# This file is part of SliceRL by M. Rossi
import numpy as np
import tensorflow as tf
from slicerl.tools import onehot, onehot_to_indices
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    InputLayer,
    Dense,
    Activation,
    Flatten,
    Reshape,
    Concatenate,
    Dropout,
    Conv1D,
    Lambda
)
from tensorflow.keras.constraints import MaxNorm
import knn

TF_DTYPE_INT = tf.int32
TF_DTYPE     = tf.float32

NP_DTYPE_INT = np.int32
NP_DTYPE     = np.float32

def float_me(x):
    return tf.constant(x, dtype=TF_DTYPE)

def int_me(x):
    return tf.constant(x, dtype=TF_DTYPE_INT)

def py_knn(points, queries, K):
    return knn.knn_batch(points, queries, K=K,omp=True)


#======================================================================
class LocSE(Layer):
    """ Class defining Local Spatial Encoding Layer. """
    #----------------------------------------------------------------------
    def __init__(self, units, K=8, dims=2, activation='relu', **kwargs):
        """
        Parameters
        ----------
            - units      : int, output channels (twice of input channels)
            - K          : int, number of nearest neighbors
            - dims       : int, point cloud spatial dimensions number
            - activation : str, MLP layer activation
        """
        super(LocSE, self).__init__(**kwargs)
        self.units      = units
        self.K          = K
        self.dims       = dims
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
        # B_, N_, n_feat = features
        # tf.debugging.assert_equal(B, B_, message=f"Batches number mismatch in input point cloud, got {B} and {B_}")
        # tf.debugging.assert_equal(N, N_, message=f"Points number mismatch in input point cloud, got {N} and {N_}")

        self.ch_dims  = self.dims*3 + 1 # point + neighbor point + relative space dims + 1 (the norm)
        self.MLP = Conv1D(self.units//2, 1, input_shape=(self.K, self.ch_dims),
                          activation=self.activation, 
                          kernel_regularizer='l2',
                          bias_regularizer='l2',
                          kernel_constraint=MaxNorm(axis=[0,1]),
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
        shape = tf.shape(pc)
        B = shape[0]
        N = shape[1]
        if cached and self._is_cached:
            rppe, n_idx = self._cache
            shape    = [None,None,self.K]
            n_idx    = tf.ensure_shape(n_idx, shape)
        else:
            inp      = [pc,pc,self.K]
            n_idx    = tf.py_function(func=py_knn, inp=inp, Tout=TF_DTYPE_INT, name='knn_search')
            shape    = [None,None,self.K]
            n_idx    = tf.ensure_shape(n_idx, shape)
            n_points = self.gather_neighbours(pc, n_idx)
            
            # point cloud with K expanded axis
            Kpc   = tf.repeat(tf.expand_dims(pc,-2), self.K, -2)
            relp  = Kpc - n_points
            norms = tf.norm(relp, ord='euclidean', axis=-1, keepdims=True, name='norm')

            # relative point position encoding
            rppe = self.cat([Kpc, n_points, relp, norms])
            self._cache     = [rppe, n_idx]
            self._is_cached = True

        # force shapes: n_feats, rppe
        # shape depends on input shape which is not known in graph mode
        rppe = tf.ensure_shape(rppe, [None,None,self.K,self.ch_dims])
        r = self.MLP(rppe)

        n_feats  = self.gather_neighbours(feats, n_idx)
        n_feats = tf.ensure_shape(n_feats,[None,None,self.K,self.units//2] )
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
        shape      = tf.shape(pc)
        B          = shape[0]
        N          = shape[1]
        dims       = shape[2]
        K          = tf.shape(n_idx)[-1]
        idx_input  = tf.reshape(n_idx, [B,-1])
        features   = tf.gather(pc, idx_input, axis=1, batch_dims=1, name='gather_neighbours')
        return tf.reshape(features, [B,N,K,dims])

#======================================================================
class AttentivePooling(Layer):
    """ Class defining Attentive Pooling Layer. """
    #----------------------------------------------------------------------
    def __init__(self, input_units, units=1, K=8, activation='relu', **kwargs):
        """
        Parameters
        ----------
            - input_units : int, number of input_units
            - units       : int, number of output units
            - activation  : str, MLP layer activation
        """
        super(AttentivePooling, self).__init__(**kwargs)
        self.input_units = input_units
        self.units       = units
        self.K           = K
        self.activation  = activation

    #----------------------------------------------------------------------
    def build(self, input_shape):
        """
        Note: the MLP kernel has size 1. This means that point features are
              not mixed across neighbours.
        """
        shape = (self.input_units, self.K)
        self.MLP_score = Conv1D(input_shape[-1], 1, input_shape=shape,
                          activation='softmax', 
                          kernel_regularizer='l2',
                          bias_regularizer='l2',
                          kernel_constraint=MaxNorm(axis=[0,1]),
                          name='attention_score_MLP')
        self.MLP_final = Conv1D(self.units, 1, input_shape=shape,
                          activation=self.activation, 
                          kernel_regularizer='l2',
                          bias_regularizer='l2',
                          kernel_constraint=MaxNorm(axis=[0,1]),
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
            "input_units" : self.input_units,
            "units"       : self.units,
            "K"           : self.K,
            "activation"  : self.activation
        })
        return config

#======================================================================
class DilatedResBlock(Layer):
    """ Class defining Dilated Residual Block. """
    #----------------------------------------------------------------------
    def __init__(self, input_units, units=1, K=8, activation='relu', **kwargs):
        """
        Parameters
        ----------
            - input_units : int, number of input_units
            - units       : int, number of output units, divisible by 4
            - K           : int, number of nearest neighbors
            - activation  : str, MLP layer activation
        """
        super(DilatedResBlock, self).__init__(**kwargs)
        self.input_units    = input_units
        self.units          = units
        self.K              = K
        self.activation     = activation

    #----------------------------------------------------------------------
    def build(self, input_shape):
        """
        Note: the MLP kernel has size 1. This means that point features are
              not mixed across neighbours.
        """
        spatial, features = input_shape
        B , N , dims   = spatial
        B_, N_, n_feat = features
        # tf.debugging.assert_equal(B, B_, message=f"Batches number mismatch in input point cloud, got {B} and {B_}")
        # tf.debugging.assert_equal(N, N_, message=f"Points number mismatch in input point cloud, got {N} and {N_}")

        self.MLP_0   = Conv1D(self.units//4, 1, input_shape=(None, self.input_units),
                              activation=self.activation, 
                              kernel_regularizer='l2',
                              bias_regularizer='l2',
                              kernel_constraint=MaxNorm(axis=[0,1]),
                              name='MLP_0')
        self.MLP_1   = Conv1D(self.units, 1, input_shape=(None, self.units//2),
                              activation=self.activation, 
                              kernel_regularizer='l2',
                              bias_regularizer='l2',
                              kernel_constraint=MaxNorm(axis=[0,1]),
                              name='MLP_2')
        self.MLP_res = Conv1D(self.units, 1, input_shape=(None, self.input_units),
                              activation=self.activation, 
                              kernel_regularizer='l2',
                              bias_regularizer='l2',
                              kernel_constraint=MaxNorm(axis=[0,1]),
                              name='MLP_res')

        self.locse_0 = LocSE(self.units//2, K=self.K, name='locse_0')
        self.locse_1 = LocSE(self.units, K=self.K, name='locse_1')

        self.att_0 = AttentivePooling(self.units//2, self.units//2, K=self.K, activation=self.activation, name='attention_0')
        self.att_1 = AttentivePooling(self.units, self.units, K=self.K, activation=self.activation, name='attention_1')

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
            "input_units" : self.input_units,
            "units"       : self.units,
            "K"           : self.K,
            "activation"  : self.activation
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
            - tf.Tensor, valid neighbours for invalid points of shape=(B,N//scale)
            - tf.Tensor, random permutation of pc points of shape=(B,N)
        """
        pc, feats = inputs

        shape = tf.shape(feats)
        B    = shape[0]
        N    = shape[1]
        dims = shape[2]

        # shape  = tf.shape(feats)
        # B_     = shape[0]
        # N_     = shape[1]
        # f_dims = shape[2]
        
        # tf.debugging.assert_equal(B, B_, message=f"Batches number mismatch in input point cloud, got {B} and {B_}")
        # tf.debugging.assert_equal(N, N_, message=f"Points number mismatch in input point cloud, got {N} and {N_}")

        rnds = tf.expand_dims(tf.random.shuffle(tf.range(N)), 0)
        # rnds = tf.TensorArray(TF_DTYPE_INT, size=B)
        # for i in tf.range(B):
        #     rnds = rnds.write(i, tf.random.shuffle(tf.range(N)))
        # rnds = rnds.stack()

        valid_idx   = rnds[:,:N//2]
        invalid_idx = rnds[:,N//2:]

        # downsample point cloud
        valid_pc = tf.gather(pc, valid_idx, axis=1, batch_dims=1, name='downsample_pc')
        valid_feats = tf.gather(feats, valid_idx, axis=1, batch_dims=1, name='downsample_features')

        # info for upsampling
        invalid_pc = tf.gather(pc, invalid_idx, axis=1, batch_dims=1)
        inp      = [valid_pc, invalid_pc, 2]
        n_idx    = tf.py_function(func=py_knn, inp=inp, Tout=TF_DTYPE_INT, name='RS_knn_search')[:,:,1]
        shape    = [None]*2
        n_idx    = tf.ensure_shape(n_idx, shape)

        return valid_pc, valid_feats, n_idx, rnds

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
    def __init__(self, input_units, units, scale=2, activation='relu', **kwargs):
        """
        Parameters
        ----------
            - scale: int, scaling factor of the input cloud
        """
        super(UpSample, self).__init__(**kwargs)
        self.input_units = input_units
        self.units       = units
        self.scale       = scale
        self.activation  = activation
    
    #----------------------------------------------------------------------
    def build(self, input_shape):
        self.MLP  = Conv1D(self.units, 1, input_shape=(None,None,self.input_units),
                          activation=self.activation, kernel_regularizer='l2',
                          bias_regularizer='l2', kernel_constraint=MaxNorm(axis=[0,1]),
                          name='MLP')

    #----------------------------------------------------------------------
    def call(self, inputs):
        """
        Parameters
        ----------
            - inputs : list of tf.Tensors,
                       feature point cloud of shape=(B,N//scale,f_dims),
                       interpolating indices of shape=(B,N//scale),
                       upsampling indices of shape=(B,N),
                       residual feature point cloud of shape=(B,N,f_dims)        
        Returns
        -------
            - tf.Tensor, features of shape=(B,N,f_dims)
        """
        feats, interpolate_idx, upsample_idx, res = inputs

        copy_feats = tf.gather(feats, interpolate_idx, axis=1,
                                       batch_dims=1, name='copy_features')

        interpolated_feats = tf.concat([feats, copy_feats], axis=1)

        shape = tf.shape(interpolated_feats)
        
        first = tf.repeat(tf.expand_dims(tf.range(shape[0]), 1), shape[1], axis=1)
        indices = tf.stack([first, upsample_idx], axis=-1)

        feats = tf.scatter_nd(indices, interpolated_feats, shape, name='upsample_features')

        feats = tf.concat([res,feats], axis=-1, name=f'res_cat')
        return self.MLP(feats)



    #----------------------------------------------------------------------
    def get_config(self):
        config = super(UpSample, self).get_config()
        config.update({
            "input_units" : self.input_units,
            "units"       : self.units,
            "scale"       : self.scale,
            "activation"  : self.activation
        })
        return config

#======================================================================
class RandLANet(Model):
    """ Class deifining RandLA-Net. """
    def __init__(self, dims=2, f_dims=2, nb_classes=128, K=16, scale_factor=2,
                 nb_layers=4, activation='relu', fc_type='conv', dropout=0.1, name='RandLA-Net', **kwargs):
        """
        Parameters
        ----------
            - nb_classes   : int, number of output classes for each point in cloud
            - K            : int, number of nearest neighbours to find
            - scale_factor : int, scale factor for down/up-sampling
            - nb_layers    : int, number of inner enocding/decoding layers
            - fc_type      : str, either 'conv' or 'dense' for the final FC
                             layers
            - dropout      : float, dropout percentage in final FC layers
        """
        super(RandLANet, self).__init__(name=name, **kwargs)

        # store args
        self.dims         = dims
        self.f_dims       = f_dims
        self.nb_classes   = nb_classes
        self.K            = K
        self.scale_factor = scale_factor
        self.nb_layers    = nb_layers
        self.activation   = activation
        self.fc_type      = fc_type
        self.dropout_perc = dropout

        # store some useful parameters
        self.fc_units = [32, 64, 128, self.nb_classes]
        self.fc_acts  = [self.activation]*3 + ['linear']
        self.latent_f_dim = 32
        self.enc_units  = [
            self.latent_f_dim*self.scale_factor**i \
                for i in range(self.nb_layers)
                          ]
        self.enc_iunits = [8] + self.enc_units[:-1]
        self.dec_units  = self.enc_units[-2::-1] + [self.fc_units[0]]
        self.fc_iunits  = self.dec_units[-1:] + self.fc_units[:-1]

        # build layers
        if self.fc_type == 'conv':
            self.fcs = [
                Conv1D(units, 16, padding='same', input_shape=(None,None,None,iunits),
                              kernel_regularizer='l2', bias_regularizer='l2',
                              kernel_constraint=MaxNorm(axis=[0,1]),
                              activation=act, name=f'fc{i+1}') \
                    for i, (iunits, units, act) in enumerate(zip(self.fc_iunits[1:], self.fc_units[1:], self.fc_acts[1:]))
            ]
            self.fcs.insert(0, Dense(self.fc_units[0],
                                     kernel_regularizer='l2', bias_regularizer='l2',
                                     activation=self.fc_acts[0], name=f'fc0') )
        elif self.fc_type == 'dense':
            self.fcs = [
                Dense(units, kernel_regularizer='l2', bias_regularizer='l2',
                      kernel_constraint=MaxNorm(), activation=act, name=f'fc{i}') \
                    for i, (units, act) in enumerate(zip(self.fc_units, self.fc_acts))
            ]
        else:
            raise NotImplementedError(f"First and final layers must be of type 'conv'|'dense', not {self.fc_type}")
        
        if self.dropout_perc:
            self.dropout = [Dropout(self.dropout_perc, name=f'dropout{i+1}') \
                        for i in range(len(self.fc_units[1:-1]))]
        else:
            self.dropout = [[] for i in range(len(self.fc_units[1:-1]))]

        self.encoding   = self.build_encoder()
        self.middle_MLP = Conv1D(self.enc_units[-1], 1,
                              activation=self.activation,
                              kernel_regularizer='l2',
                              bias_regularizer='l2',
                              name='MLP')
        self.decoding   = self.build_decoder()
        
    #----------------------------------------------------------------------
    def build_encoder(self):
        self.encoder = [
            DilatedResBlock(input_units=iunits, units=units, K=self.K, activation=self.activation, name=f'DRB{i}') \
                                for i, (iunits, units) in enumerate(zip(self.enc_iunits, self.enc_units))
                       ]
        self.RSs = [
            RandomSample(self.scale_factor, name=f'RS{i}') \
                for i in range(self.nb_layers)
                   ]

    #----------------------------------------------------------------------
    def build_decoder(self):
        self.dec_iunits = self.enc_units[::-1]
        
        self.USs = [
            UpSample(iunits, units, self.scale_factor, activation=self.activation, name=f'US{i}') \
                for i, (iunits, units) in enumerate(zip(self.dec_iunits, self.dec_units))
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
            tf.Tensor, output tensor of shape=(B,N,nb_classes)
        """
        pc, feats = inputs

        residuals = []
        n_idxs    = []
        rndss     = []

        feats = self.fcs[0](feats)
        residuals.append(feats)

        for drb, rs  in zip(self.encoder, self.RSs):
            feats = drb( [pc, feats] )
            pc, feats, n_idx, rnds = rs([pc, feats])
            n_idxs.append(n_idx)
            rndss.append(rnds)
            residuals.append(feats)

        feats = self.middle_MLP(feats)
        for i, (us, interp, ups, res) in enumerate(zip(self.USs, n_idxs[::-1], rndss[::-1], residuals[-2::-1])):
            feats = us([feats, interp, ups, res])

        for fc, do in zip(self.fcs[1:-1], self.dropout):
            feats = fc(feats)
            if self.dropout_perc:
                feats = do(feats)

        return self.fcs[-1](feats) # logits

    #----------------------------------------------------------------------
    def model(self):
        pc = Input(shape=(None, self.dims), name='pc')
        feats = Input(shape=(None, self.f_dims), name='feats')
        return Model(inputs=[pc, feats], outputs=self.call([pc,feats]), name=self.name)

    #----------------------------------------------------------------------
    def get_prediction(self, inputs):
        """
        Predict over a iterable of inputs

        Parameters
        ----------
            - inputs : list, elements of shape [(1,N,dims), (N,f_dims)]
        
        Returns
        -------
            - pred  : list, output list containing np.array of predictions,
                      each of shape=[(B,N)]
            - probs : list, output list containing np.array of class probabilities,
                      each of shape=[(B,N,nb_classes)]
        """
        pred   = []
        probs = []
        for inp in inputs:
            out = self.predict_on_batch(inp)
            pred.append(onehot_to_indices(out))
            probs.append( tf.nn.softmax(out, axis=-1).numpy() )
        return pred, probs