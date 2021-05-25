from slicerl.config import NP_DTYPE_INT
from slicerl.tools import bfs

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import (
    Input,
    Reshape,
    Concatenate,
    Conv1D,
    BatchNormalization
)
from tensorflow.keras.constraints import MaxNorm

#======================================================================
class LocSE(Layer):
    """ Class defining Local Spatial Encoding Layer. """
    #----------------------------------------------------------------------
    def __init__(self, units, K=8, dims=2, activation='relu', use_bias=True,
                 use_bnorm=False, **kwargs):
        """
        Parameters
        ----------
            - units      : int, output channels (twice of input channels)
            - K          : int, number of nearest neighbors
            - dims       : int, point cloud spatial dimensions number
            - activation : str, MLP layer activation
            - use_bias   : bool, wether to use bias in MLPs
            - use_bnorm  : bool, wether to use batchnormalization            
        """
        super(LocSE, self).__init__(**kwargs)
        self.units      = units
        self.K          = K
        self.dims       = dims
        self.activation = activation
        self._cache     = None
        self._is_cached = False
        self.use_bias   = use_bias
        self.use_bnorm  = use_bnorm

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
        self.ch_dims  = (self.dims + 1)*(1+self.K) + self.dims # point + neighbor point + relative space dims + 1 (the norm)
        self.MLP = Conv1D(self.units, 1, input_shape=(1+self.K, self.ch_dims),
                          activation=self.activation,
                          use_bias=self.use_bias,
                          # kernel_regularizer='l2',
                          # bias_regularizer='l2',
                          kernel_constraint=MaxNorm(axis=[0,1]),
                          name='MLP')
        self.cat = Concatenate(axis=-1, name='cat')
        if self.use_bnorm:
            self.bnorm = BatchNormalization()

    #----------------------------------------------------------------------
    def call(self, inputs, use_cache=False):
        """
        Layer forward pass.

        Parameters
        ----------
            - inputs     : list of tf.Tensors, tensors describing spatial and
                           feature point cloud of shapes=[(B,N,1+K,dims), (B,N,1+K,f_dims)]

        Returns
        -------
            tf.Tensor of shape=(B,N,1+K,2*f_dims)
        """
        pc, feats = inputs
        if use_cache and self._is_cached:
            rppe   = self._cache
        else:
            # relative positions between neighbours
            rpbns = []
            for i in range(1+self.K):
                current = pc[:,:,i:i+1]
                diff = current - pc
                norms = tf.norm(pc, ord='euclidean', axis=-1, keepdims=True, name='norm')
                rpbn = tf.concat([diff, norms], axis=-1)
                rpbns.append(rpbn)

            # relative point position encoding
            rppe = self.cat([pc] + rpbns)

            # save cache
            self._cache     = rppe
            self._is_cached = True

        # force shapes: n_feats, rppe
        # shape depends on input shape which is not known in graph mode
        # MLP call cannot be included in the cache because each MLP in the
        # network has different weights
        rppe = tf.ensure_shape(rppe, [None,None,1+self.K,self.ch_dims])
        r = self.MLP(rppe)

        if self.use_bnorm:
            r = self.bnorm(r)

        return self.cat([r, feats])

    #----------------------------------------------------------------------
    def get_config(self):
        config = super(LocSE, self).get_config()
        config.update({
            "units"      : self.units,
            "K"          : self.K,
            "activation" : self.activation,
            "use_bias"   : self.use_bias,
            "use_bnorm"  : self.use_bnorm
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
    def __init__(self, dh, do, K, ds=None, dims=2, f_dims=2,
                 activation='relu', use_bias=True, use_cache=True, use_bnorm=False,
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
            - activation : str, MLP layer activation
            - use_bias   : bool, wether to use bias in MLPs
            - use_bnorm  : bool, wether to use batchnormalization
        """
        super(SEAC, self).__init__(name=name, **kwargs)
        self.dh         = dh
        self.do         = do
        self.ds         = ds if ds else dh
        self.da         = self.ds + self.dh
        self.K          = K
        self.dims       = dims
        self.f_dims     = f_dims
        self.activation = activation
        self.use_bias   = use_bias
        self.use_cache  = use_cache
        self.use_bnorm  = use_bnorm

        self.locse = LocSE(
                        self.ds,
                        K=self.K,
                        use_bias=self.use_bias,
                        use_bnorm=self.use_bnorm,
                        name='locse')

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
                           name='skip_conv')
        if self.use_bnorm:
            self.bns = [BatchNormalization() for i in range(2)]

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
        if self.use_bnorm:
            reshaped = self.bns[0](reshaped)
        cat = self.cat([pc[:,:,0], reshaped])
        res = self.reshape1( self.conv(cat) )
        skip = self.skip_conv( self.cat([feats, res]) )
        if self.use_bnorm:
            skip = self.bns[1](skip)
        return skip

    #----------------------------------------------------------------------
    def get_config(self):
        config = super(SEAC, self).get_config()
        config.update({
            "ds"         : self.ds,
            "da"         : self.da,
            "dh"         : self.dh,
            "do"         : self.do,
            "K"          : self.K,
            "activation" : self.activation,
            "use_bias"   : self.use_bias,
            "use_cache"  : self.use_cache,
            "use_bnorm"  : self.use_bnorm
         })
        return config

#======================================================================
class Predictions:
    """ Utility class to return RandLA-Net predictions. """
    def __init__(self, graphs, status, preds, slices):
        """
        Parameters
        ----------
            - graph  : list, KNN graphs which is list with elements of shape=(nb_neighs)
            - status : list, each element is a np.array with shape=(N)
            - preds  : list, each element is a np.array with shape=(N,1+K)
            - slices : list, of sets with decreasing size containing
        """
        self.graphs = graphs
        self.status = status
        self.preds  = preds
        self.slices = slices

    #----------------------------------------------------------------------
    def get_graph(self, index):
        """
        Returns the i-th graph.

        Parameters
        ----------
            - index : int, index in prediction list

        Returns
        -------
            - np.array: graph at index i of shape=(N,nb_neighs,2)
        """
        return self.graphs[index]
    #----------------------------------------------------------------------
    def get_status(self, index):
        """
        Returns the slice state for each hit in the i-th graph.
        Parameters
        ----------
            - index : int, index in status list

        Returns
        -------
            - np.array: status at index i of shape=(N,)
        """
        return self.status[index]

    #----------------------------------------------------------------------
    def get_preds(self, index):
        """
        Returns the predictions for each possible edge in the i-th graph.
        Range is [0,1].

        Parameters
        ----------
            - index : int, index in preds list

        Returns
        -------
            - np.array: status at index i of shape=(N,1+K)
        """
        return self.preds[index]

    #----------------------------------------------------------------------
    def get_slices(self, index):
        """
        Returns the slices: each slice contains the calohits indices inside the
        slice set.

        Parameters
        ----------
            - index : int, index in slice list

        Returns
        -------
            - list: of set objects with calohit indices
        """
        return self.slices[index]

#======================================================================
class AbstractNet(Model):
    def __init__(self, name, **kwargs):
        super(AbstractNet, self).__init__(name=name, **kwargs)

    #----------------------------------------------------------------------
    def model(self):
        pc = Input(shape=(None, 1+self.K, self.dims), name='pc')
        feats = Input(shape=(None, 1+self.K, self.f_dims), name='feats')
        return Model(inputs=[pc, feats], outputs=self.call([pc,feats]), name=self.name)

    #----------------------------------------------------------------------
    def get_prediction(self, inputs, knn_idxs, threshold=0.5):
        """
        Predict over a iterable of inputs

        Parameters
        ----------
            - inputs    : list, elements of shape [(1,N,1+K,dims), (1,N,1+K,f_dims)]
            - knn_idxs  : list, elements of shape=(N,1+K)
            - threshold : float, classification threshold

        Returns
        -------
            Predictions object
        """
        status = []
        graphs = []
        preds  = []
        all_slices = []
        for inp, knn_idx in zip(inputs, knn_idxs):
            # predict hits connections
            pred = self.predict_on_batch(inp)

            preds.append(pred[0])
            graph = [set(node[p > threshold]) for node, p in zip(knn_idx[0], pred[0])]
            graphs.append( graph )

            # DFS (depth first search)
            visited = set() # the all visited set
            slices = []
            for node in range(len(graph)):
                if node in visited:
                    continue
                slice = set() # the current slice only
                bfs(slice, visited, node, graph)
                slices.append(slice)

            N = inp[0].shape[1]
            sorted_slices = sorted(slices, key=len, reverse=True)
            all_slices.append(sorted_slices)
            state = np.zeros(N)
            for i, slice in enumerate(sorted_slices):
                state[np.array(list(slice), dtype=NP_DTYPE_INT)] = i
            status.append(state)

        return Predictions(graphs, status, preds, all_slices)
