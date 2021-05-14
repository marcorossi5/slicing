from slicerl.tools import onehot_to_indices, m_lin_fit_tf, pearson_distance_tf
from slicerl.config import TF_DTYPE_INT

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import (
    Input,
    Activation,
    Reshape,
    Concatenate,
    Conv1D
)
from tensorflow.keras.constraints import MaxNorm

import knn

def py_knn(points, queries, K):
    return knn.knn_batch(points, queries, K=K,omp=True)

#======================================================================
class LocSE(Layer):
    """ Class defining Local Spatial Encoding Layer. """
    #----------------------------------------------------------------------
    def __init__(self, units, K=8, dims=2, activation='relu', use_bias=True, **kwargs):
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
        self.use_bias   = use_bias

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

        self.ch_dims  = (self.dims + 1)*(self.K + 1) + self.dims # point + neighbor point + relative space dims + 1 (the norm)
        self.MLP = Conv1D(self.units//2, 1, input_shape=(self.K, self.ch_dims),
                          activation=self.activation,
                          use_bias=self.use_bias,
                          # kernel_regularizer='l2',
                          # bias_regularizer='l2',
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
            tf.Tensor of shape=(B,N,2)
        """
        pc, feats = inputs
        shape = tf.shape(pc)
        B = shape[0]
        N = shape[1]
        if cached and self._is_cached:
            rppe, n_idx, ggf = self._cache
            shape    = [None,None,self.K+1]
            n_idx    = tf.ensure_shape(n_idx, shape)
        else:
            inp      = [pc,pc,self.K+1]
            n_idx    = tf.py_function(func=py_knn, inp=inp, Tout=TF_DTYPE_INT, name='knn_search')
            shape    = [None,None,self.K+1]
            n_idx    = tf.ensure_shape(n_idx, shape)
            n_points = self.gather_neighbours(pc, n_idx)

            # relative positions between neighbours
            rpbns = []
            for i in range(self.K+1):
                current = n_points[:,:,i:i+1]
                diff = current - n_points
                norms = tf.norm(n_points, ord='euclidean', axis=-1, keepdims=True, name='norm')
                rpbn = tf.concat([diff, norms], axis=-1)
                rpbns.append(rpbn)

            # relative point position encoding
            rppe = self.cat([n_points] + rpbns)

            # global graph feat
            ggf = tf.concat([m_lin_fit_tf(n_points), pearson_distance_tf(n_points)], axis=-1)

            # save cache
            self._cache     = [rppe, n_idx, ggf]
            self._is_cached = True

        # force shapes: n_feats, rppe
        # shape depends on input shape which is not known in graph mode
        # MLP call cannot be included in the cache because each MLP in the
        # network has different weights
        rppe = tf.ensure_shape(rppe, [None,None,self.K+1,self.ch_dims])
        r = self.MLP(rppe)

        n_feats  = self.gather_neighbours(feats, n_idx)
        n_feats = tf.ensure_shape( n_feats, [None,None,self.K+1,self.units//2] )

        ggf = tf.ensure_shape( ggf, [None,None,1,2] )
        return self.cat([n_feats, r]), ggf

    #----------------------------------------------------------------------
    def get_config(self):
        config = super(LocSE, self).get_config()
        config.update({
            "units"      : self.units,
            "K"          : self.K,
            "activation" : self.activation,
            "use_bias"   : self.use_bias
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
    def __init__(self, input_units, units=1, K=8, activation='relu', use_bias=True, **kwargs):
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
        self.use_bias    = use_bias

    #----------------------------------------------------------------------
    def build(self, input_shape):
        """
        Note: the MLP kernel has size 1. This means that point features are
              not mixed across neighbours.
        """
        input_shape, _ = input_shape # the second item is gg_shape
        shape = (self.input_units, self.K+1)
        self.MLP_score = Conv1D(input_shape[-1], 1, input_shape=shape,
                          activation='softmax',
                          use_bias=self.use_bias,
                          # kernel_regularizer='l2',
                          # bias_regularizer='l2',
                          kernel_constraint=MaxNorm(axis=[0,1]),
                          name='attention_score')
        self.MLP_final = Conv1D(self.units, 1, input_shape=shape,
                          activation=self.activation,
                          use_bias=self.use_bias,
                          # kernel_regularizer='l2',
                          # bias_regularizer='l2',
                          kernel_constraint=MaxNorm(axis=[0,1]),
                          name='shared_MLP')
        self.reshape   = Reshape((-1, self.units), name='reshape')
        self.cat       = Concatenate(axis=-1, name='cat')

    #----------------------------------------------------------------------
    def call(self, inputs):
        """
        Layer forward pass.

        Parameters
        ----------
            - inputs: list, containing output of LocSE layer of shapes=[(B,N,dims),(B,N,2)]

        Returns
        -------
            tf.Tensor of shape=(B,N,units)
        """
        n_feats, ggf = inputs
        scores = self.MLP_score(n_feats)
        attention = tf.math.reduce_sum(n_feats*scores, axis=-2, keepdims=True)
        
        # add some geometric inspired quantities about the neighborhood
        cat = self.cat([attention,ggf])

        return self.reshape( self.MLP_final(cat) )

    #----------------------------------------------------------------------
    def get_config(self):
        config = super(AttentivePooling, self).get_config()
        config.update({
            "input_units" : self.input_units,
            "units"       : self.units,
            "K"           : self.K,
            "activation"  : self.activation,
            "use_bias"   : self.use_bias
        })
        return config

#======================================================================
class DilatedResBlock(Layer):
    """ Class defining Dilated Residual Block. """
    #----------------------------------------------------------------------
    def __init__(self, input_units, units=1, K=8, activation='relu', use_bias=True, all_cached=False, **kwargs):
        """
        Parameters
        ----------
            - input_units : int, number of input_units
            - units       : int, number of output units, divisible by 4
            - K           : int, number of nearest neighbors
            - activation  : str, MLP layer activation
            - all_cached  : bool, wether to run KNN or use cached
        """
        super(DilatedResBlock, self).__init__(**kwargs)
        self.input_units    = input_units
        self.units          = units
        self.K              = K
        self.activation     = activation
        self.use_bias       = use_bias
        self.all_cached     = all_cached

        self.locse_0 = LocSE(self.units//2, K=self.K, use_bias=self.use_bias, name='locse_0')
        self.locse_1 = LocSE(self.units, K=self.K, use_bias=self.use_bias, name='locse_1')

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
                              use_bias=self.use_bias,
                              # kernel_regularizer='l2',
                              # bias_regularizer='l2',
                              kernel_constraint=MaxNorm(axis=[0,1]),
                              name='MLP_0')
        self.MLP_1   = Conv1D(self.units, 1, input_shape=(None, self.units//2),
                              activation=self.activation,
                              use_bias=self.use_bias,
                              # kernel_regularizer='l2',
                              # bias_regularizer='l2',
                              kernel_constraint=MaxNorm(axis=[0,1]),
                              name='MLP_2')
        self.MLP_res = Conv1D(self.units, 1, input_shape=(None, self.input_units),
                              activation=self.activation,
                              use_bias=self.use_bias,
                              # kernel_regularizer='l2',
                              # bias_regularizer='l2',
                              kernel_constraint=MaxNorm(axis=[0,1]),
                              name='MLP_res')

        self.att_0 = AttentivePooling(
                self.units//2, self.units//2, K=self.K, activation=self.activation,
                use_bias=self.use_bias, name='attention_0'
                                     )
        self.att_1 = AttentivePooling(
                self.units, self.units, K=self.K, activation=self.activation,
                use_bias=self.use_bias, name='attention_1'
                                     )

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
        x = self.att_0( self.locse_0([pc, x], cached=self.all_cached) )

        # store cache in locse_1 to skip knn building
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
            "activation"  : self.activation,
            "use_bias"    : self.use_bias,
            "all_cached"  : self.all_cached
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
        # B    = shape[0]
        N    = shape[1]
        # dims = shape[2]

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
    def __init__(self, input_units, units, scale=2, activation='relu', use_bias=True, **kwargs):
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
        self.use_bias    = use_bias

    #----------------------------------------------------------------------
    def build(self, input_shape):
        self.MLP  = Conv1D(self.units, 1, input_shape=(None,None,self.input_units),
                          activation=self.activation,
                          use_bias=self.use_bias,
                          # kernel_regularizer='l2',
                          # bias_regularizer='l2',
                          kernel_constraint=MaxNorm(axis=[0,1]),
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
            "activation"  : self.activation,
            "use_bias"    : self.use_bias
        })
        return config

#======================================================================
class Predictions:
    """ Utility class to return RandLA-Net predictions. """
    def __init__(self, predictions):
        """
        Parameters
        ----------
            - predictions : list, each element is a tf.Tensor with
                            shape=(N,nb_classes)
        """
        self.predictions = [onehot_to_indices(pred.numpy()) for pred in predictions]
        self.probs = [tf.nn.softmax(pred, axis=-1).numpy() for pred in predictions]

    #----------------------------------------------------------------------
    def get_pred(self, index):
        """
        Parameters
        ----------
            - index : int, index in prediction list

        Returns
        -------
            - np.array: prediction at index i of shape=(N,)
        """
        return self.predictions[index]

    #----------------------------------------------------------------------
    def get_probs(self, index):
        """
        Parameters
        ----------
            - index : int, index in probs list

        Returns
        -------
            - np.array: prediction at index i of shape=(N,nb_classes)
        """
        return self.probs[index]

#======================================================================
class AbstractNet(Model):
    def __init__(self, name, **kwargs):
        super(AbstractNet, self).__init__(name=name, **kwargs)

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
            - inputs : list, elements of shape [(1,N,dims), (1,N,f_dims)]

        Returns
        -------
            Predictions object
        """
        predictions = [ tf.squeeze(self.predict_on_batch(inp), 0) \
                            for inp in inputs]
        return Predictions(predictions)