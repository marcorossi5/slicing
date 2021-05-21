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
    Conv1D,
    BatchNormalization
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
        spatial, _ = input_shape
        B , N , _   = spatial

        self.ch_dims  = (self.dims + 1)*(1+self.K) + self.dims # point + neighbor point + relative space dims + 1 (the norm)
        self.MLP = Conv1D(self.units, 1, input_shape=(1+self.K, self.ch_dims),
                          activation=self.activation,
                          use_bias=self.use_bias,
                          # kernel_regularizer='l2',
                          # bias_regularizer='l2',
                          kernel_constraint=MaxNorm(axis=[0,1]),
                          name='MLP')
        self.cat = Concatenate(axis=-1, name='cat')

    #----------------------------------------------------------------------
    def call(self, inputs, use_cache=False):
        """
        Layer forward pass.

        Parameters
        ----------
            - inputs     : list of tf.Tensors, tensors describing spatial and
                           feature point cloud of shapes=[(B,N,dims), (B,N,1+K,f_dims)]

        Returns
        -------
            tf.Tensor of shape=(B,N,1+K,2*f_dims)
        """
        pc, feats = inputs
        shape = tf.shape(pc)
        B = shape[0]
        N = shape[1]
        if use_cache and self._is_cached:
            rppe, n_idx = self._cache
            shape    = [None,None,1+self.K]
            n_idx    = tf.ensure_shape(n_idx, shape)
            n_feats  = feats
        else:
            inp      = [pc,pc,1+self.K]
            n_idx    = tf.py_function(func=py_knn, inp=inp, Tout=TF_DTYPE_INT, name='knn_search')
            shape    = [None,None,self.K+1]
            n_idx    = tf.ensure_shape(n_idx, shape)
            n_points = self.gather_neighbours(pc, n_idx)

            n_feats  = self.gather_neighbours(feats, n_idx)
            n_feats = tf.ensure_shape( n_feats, [None,None,1+self.K,self.units] )

            # relative positions between neighbours
            rpbns = []
            for i in range(1+self.K):
                current = n_points[:,:,i:i+1]
                diff = current - n_points
                norms = tf.norm(n_points, ord='euclidean', axis=-1, keepdims=True, name='norm')
                rpbn = tf.concat([diff, norms], axis=-1)
                rpbns.append(rpbn)

            # relative point position encoding
            rppe = self.cat([n_points] + rpbns)

            # save cache
            self._cache     = [rppe, n_idx]
            self._is_cached = True

        # force shapes: n_feats, rppe
        # shape depends on input shape which is not known in graph mode
        # MLP call cannot be included in the cache because each MLP in the
        # network has different weights
        rppe = tf.ensure_shape(rppe, [None,None,1+self.K,self.ch_dims])
        r = self.MLP(rppe)

        return self.cat([r, n_feats])

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
class SEAC(Layer):
    """ Class defining Spatial Encoding Attention Convolutional Layer. """
    #----------------------------------------------------------------------
    def __init__(self, dh, do, K, ds=None, dims=2, f_dims=2, skip_link=True,
                 activation='relu', use_bias=True, use_cache=True,
                 name='seac', **kwargs):
        """
        Parameters
        ----------
            - dh         : int, number of hidden feature dimensions
            - do         : int, number of output dimensions
            - ds         : int, number of spatial encoding output dimensions
            - K          : int, number of nearest neighbors
            - dims       : int, point cloud spatial dimensions
            - f_dims     : int, point cloud feature dimensions
            - skip_link  : bool, wether to convolute inputs with outpus
            - activation : str, MLP layer activation
            - use_bias   : wether to use bias in MLPs
        """
        super(SEAC, self).__init__(name=name, **kwargs)
        self.dh         = dh
        self.do         = do
        self.ds         = ds if ds else dh
        self.da         = self.ds + self.dh
        self.K          = K
        self.dims       = dims
        self.f_dims     = f_dims
        self.skip_link  = skip_link
        self.activation = activation
        self.use_bias   = use_bias
        self.use_cache  = use_cache

        self.locse = LocSE(self.ds, K=self.K, use_bias=self.use_bias, name='locse')

    #----------------------------------------------------------------------
    def build(self, input_shape):
        """
        Note: the MLP kernel has size 1. This means that point features are
              not mixed across neighbours.
        """
        shape = (None, 1 + self.K, self.da)
        self.att  = Conv1D(self.da, 1, input_shape=shape,
                          activation='softmax',
                          use_bias=self.use_bias,
                          kernel_constraint=MaxNorm(axis=[0,1]),
                          name='attention_score')
        self.reshape0 = Reshape((-1,(1+self.K)*self.da), name='reshape0')
        shape = (None, self.f_dims + self.dims + (1+self.K)*self.da)
        self.conv = Conv1D((1+self.K)*self.do, 1, input_shape=shape,
                           activation=self.activation,
                           use_bias=self.use_bias,
                           kernel_constraint=MaxNorm(axis=[0,1]),
                           name='conv')
        self.cat = Concatenate(axis=-1, name='cat')
        self.reshape1 = Reshape((-1,1+self.K, self.do), name='reshape1')

        shape = (None, 1 + self.K, self.do + input_shape[-1][-1])
        self.skip_conv = Conv1D(self.do, 1, input_shape=shape,
                           activation=self.activation,
                           use_bias=self.use_bias,
                           kernel_constraint=MaxNorm(axis=[0,1]),
                           name='conv')

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
            tf.Tensor of shape=(B,N,K,do)
        """
        pc, feats = inputs
        se = self.locse([pc, feats], use_cache=self.use_cache)
        attention_score = self.att(se) * se
        reshaped = self.reshape0(attention_score)
        cat = self.cat([pc, reshaped])
        res = self.reshape1( self.conv(cat) )
        if self.skip_link:
            res = self.skip_conv(self.cat([feats, res]))
        return res

    #----------------------------------------------------------------------
    def get_config(self):
        config = super(SEAC, self).get_config()
        config.update({
            "ds"         : self.ds,
            "da"         : self.da,
            "dh"         : self.dh,
            "do"         : self.do,
            "K"          : self.K,
            "skip_link"  : self.skip_link,
            "activation" : self.activation,
            "use_bias"   : self.use_bias,
            "use_cache"  : self.use_cache
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