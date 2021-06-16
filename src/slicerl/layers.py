from slicerl.config import NP_DTYPE_INT, EPS_TF
from slicerl.tools import bfs

import numpy as np
import tensorflow as tf
import tensorflow.math as tfmath
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import (
    Input,
    Reshape,
    Concatenate,
    Conv1D,
)
from tensorflow.keras.constraints import MaxNorm

fn = lambda x, y: y / (tfmath.abs(x) + EPS_TF) * tfmath.sign(x)


def encode(inputs, layers, loop=True):
    """
    Encoding forward pass. Implements residual connections.

    Parameters
    ----------
        - inputs : tf.Tensor, input tensor
        - layers : list, of tf.keras.layers.Layer subclasses

    Returns
    -------
        tf.Tensor, result of the computation
    """
    result = layers[0](inputs + tf.reduce_sum(inputs, axis=-1, keepdims=True))

    if loop:
        for layer in layers[1:]:
            residual = layer(
                result + tf.reduce_sum(result, axis=-1, keepdims=True)
            )
            result = result + residual
    return result


# ======================================================================
class LocSE(Layer):
    """ Class defining Local Spatial Encoding Layer. """

    # ----------------------------------------------------------------------
    def __init__(
        self,
        units,
        K=8,
        dims=2,
        nb_layers=3,
        activation="relu",
        use_bias=True,
        **kwargs,
    ):
        """
        Parameters
        ----------
            - units      : int, output channels (twice of input channels)
            - K          : int, number of nearest neighbors
            - dims       : int, point cloud spatial dimensions number
            - nb_layers  : int, number of shared MLP layers for each encoding
            - activation : str, MLP layer activation
            - use_bias   : bool, wether to use bias in MLPs
        """
        super(LocSE, self).__init__(**kwargs)
        self.units = units
        self.K = K
        self.dims = dims
        self.nb_layers = nb_layers
        self.activation = activation
        self._cache = None
        self._is_cached = False
        self.use_bias = use_bias

        self.enc_with_loop = True if self.nb_layers > 1 else False

    # ----------------------------------------------------------------------
    @property
    def cache(self):
        return self._cache

    # ----------------------------------------------------------------------
    @cache.setter
    def cache(self, cache):
        """
        Parameters
        ----------
            - cache : list, [n_points, n_feats] knn cache
        """
        self._cache = cache
        self._is_cached = True

    # ----------------------------------------------------------------------
    def build(self, input_shape):
        """
        Note: the MLP kernel has size 1. This means that point features are
              not mixed across neighbours.

        Parameters
        ----------
            input_shape : list of tf.TensorShape, first is (B,N,spatial dims),
                          second is (B,N,feature number)

        """
        self.ch_dims = (
            self.dims * 2 + 2 + 2
        )  # point + relative space dims + norm + angles + directions

        self.pc_enc = [
            Conv1D(
                self.units,
                3,
                padding="same",
                input_shape=(1 + self.K, self.dims),
                activation=self.activation,
                use_bias=self.use_bias,
                name=f"pc_enc{i}",
            )
            for i in range(1, self.nb_layers + 1)
        ]

        self.rel_pos_enc = [
            Conv1D(
                self.units,
                3,
                padding="same",
                input_shape=(1 + self.K, self.dims),
                activation=self.activation,
                use_bias=self.use_bias,
                name=f"rel_pos_enc{i}",
            )
            for i in range(1, self.nb_layers + 1)
        ]

        self.angles_enc = [
            Conv1D(
                self.units,
                3,
                padding="same",
                input_shape=(1 + self.K, 1),
                activation=self.activation,
                use_bias=self.use_bias,
                name=f"angles_enc{i}",
            )
            for i in range(1, self.nb_layers + 1)
        ]

        self.norms_enc = [
            Conv1D(
                self.units,
                3,
                padding="same",
                input_shape=(1 + self.K, 1),
                activation=self.activation,
                use_bias=self.use_bias,
                name=f"norms_enc{i}",
            )
            for i in range(1, self.nb_layers + 1)
        ]

        self.MLP = Conv1D(
            self.units,
            1,
            padding="same",
            input_shape=(1 + self.K, self.ch_dims),
            activation=self.activation,
            use_bias=self.use_bias,
            name=f"MLP",
        )

        self.cat = Concatenate(axis=-1, name="cat")

    # ----------------------------------------------------------------------
    def call(self, inputs, use_cache=False):
        """
        Layer forward pass.

        Parameters
        ----------
            - inputs : list of tf.Tensors, tensors describing spatial and
                       feature point cloud of shapes=[(B,N,1+K,dims), (B,N,1+K,f_dims)]

        Returns
        -------
            tf.Tensor of shape=(B,N,1+K,2*f_dims)
        """
        pc, feats = inputs
        if use_cache and self._is_cached:
            rppe = self._cache
        else:
            # relative positions between neighbours
            current = pc[:, :, :1]
            diff = current[..., :2] - pc[..., :2]
            norms = tf.norm(
                pc[..., :2],
                ord="euclidean",
                axis=-1,
                keepdims=True,
                name="norm",
            )

            diff_norm = tf.norm(
                diff,
                ord="euclidean",
                axis=-1,
                keepdims=True,
                name="norm",
            )

            current_norm = tf.norm(
                current,
                ord="euclidean",
                axis=-1,
                keepdims=True,
                name="norm",
            )

            num = tf.reduce_sum(diff * current[...,2:4], axis=-1, keepdims=True)
            den = diff_norm * current_norm + EPS_TF

            angles = 1 - tfmath.abs(num / den) # angle cosine

            rpbn = tf.concat([diff, norms, angles], axis=-1)

            # relative point position encoding
            rppe = self.cat([pc] + [rpbn])

            # save cache
            self._cache = rppe
            self._is_cached = True

        # force shapes: n_feats, rppe
        # shape depends on input shape which is not known in graph mode
        # MLP call cannot be included in the cache because each MLP in the
        # network has different weights
        rppe = tf.ensure_shape(rppe, [None, None, 1 + self.K, self.ch_dims])

        pos = rppe[..., :2]
        rel_pos = rppe[..., 2:4]
        norms = rppe[..., 4:5]
        angles = rppe[..., 5:]

        pos = encode(pos, self.pc_enc, loop=self.enc_with_loop)
        rel_pos = encode(rel_pos, self.rel_pos_enc, loop=self.enc_with_loop)
        norms = encode(norms, self.norms_enc, loop=self.enc_with_loop)
        angles = encode(angles, self.angles_enc, loop=self.enc_with_loop)

        # r = self.MLP(self.cat([pos, rel_pos, norms, angles]))
        r = self.MLP(self.cat([norms, angles]))

        return self.cat([r, feats])

    # ----------------------------------------------------------------------
    def get_config(self):
        config = super(LocSE, self).get_config()
        config.update(
            {
                "units": self.units,
                "K": self.K,
                "nb_layers": self.nb_layers,
                "activation": self.activation,
                "use_bias": self.use_bias,
            }
        )
        return config

    # ----------------------------------------------------------------------
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
        shape = tf.shape(pc)
        B = shape[0]
        N = shape[1]
        dims = shape[2]
        K = tf.shape(n_idx)[-1]
        idx_input = tf.reshape(n_idx, [B, -1])
        features = tf.gather(
            pc, idx_input, axis=1, batch_dims=1, name="gather_neighbours"
        )
        return tf.reshape(features, [B, N, K, dims])


# ======================================================================
class SEAC(Layer):
    """ Class defining Spatial Encoding Attention Convolutional Layer. """

    # ----------------------------------------------------------------------
    def __init__(
        self,
        dh,
        do,
        K,
        ds=None,
        dims=2,
        f_dims=2,
        locse_nb_layers=3,
        activation="relu",
        use_bias=True,
        use_cache=True,
        name="seac",
        **kwargs,
    ):
        """
        Parameters
        ----------
            - dh              : int, number of hidden feature dimensions
            - do              : int, number of output dimensions
            - ds              : int, number of spatial encoding output dimensions
            - K               : int, number of nearest neighbors
            - dims            : int, point cloud spatial dimensions
            - f_dims          : int, point cloud feature dimensions
            - locse_nb_layers : int, number of hidden layers in LocSE block
            - activation      : str, MLP layer activation
            - use_bias        : bool, wether to use bias in MLPs
        """
        super(SEAC, self).__init__(name=name, **kwargs)
        self.dh = dh
        self.do = do
        self.ds = ds if ds else dh
        self.da = self.ds + self.dh
        self.K = K
        self.dims = dims
        self.f_dims = f_dims
        self.activation = activation
        self.use_bias = use_bias
        self.use_cache = use_cache

        self.locse = LocSE(
            self.ds,
            K=self.K,
            nb_layers=locse_nb_layers,
            use_bias=self.use_bias,
            name="locse",
        )

    # ----------------------------------------------------------------------
    def build(self, input_shape):
        """
        Note: the MLP kernel has size 1. This means that point features are
              not mixed across neighbours.
        """
        shape = (None, 1 + self.K, self.da)
        self.att = Conv1D(
            self.da,
            1,
            input_shape=shape,
            activation="softmax",
            use_bias=self.use_bias,
            kernel_constraint=MaxNorm(axis=[0, 1]),
            name="attention_score",
        )
        self.reshape0 = Reshape((-1, (1 + self.K) * self.da), name="reshape0")
        shape = (None, self.f_dims + self.dims + (1 + self.K) * self.da)

        self.conv_block_0 = [
            Conv1D(
                10,
                3,
                padding="same",
                input_shape=shape,
                activation=self.activation,
                use_bias=self.use_bias,
                kernel_constraint=MaxNorm(axis=[0, 1]),
                name=f"conv_block0_{i}",
            )
            for i in range(5)
        ]

        self.conv_block_1 = [
            Conv1D(
                10,
                3,
                padding="same",
                input_shape=shape,
                activation=self.activation,
                use_bias=self.use_bias,
                kernel_constraint=MaxNorm(axis=[0, 1]),
                name=f"conv_block1_{i}",
            )
            for i in range(5)
        ]

        self.conv_block_2 = [
            Conv1D(
                10,
                3,
                padding="same",
                input_shape=shape,
                activation=self.activation,
                use_bias=self.use_bias,
                kernel_constraint=MaxNorm(axis=[0, 1]),
                name=f"conv_block2_{i}",
            )
            for i in range(5)
        ]

        self.conv = Conv1D(
            (1 + self.K) * self.do,
            1,
            input_shape=shape,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_constraint=MaxNorm(axis=[0, 1]),
            name="conv",
        )
        self.cat = Concatenate(axis=-1, name="cat")
        self.reshape1 = Reshape((-1, 1 + self.K, self.do), name="reshape1")

        shape = (None, 1 + self.K, self.do + input_shape[-1][-1])
        self.skip_conv = Conv1D(
            self.do,
            1,
            input_shape=shape,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_constraint=MaxNorm(axis=[0, 1]),
            name="skip_conv",
        )

    # ----------------------------------------------------------------------
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
        cat = self.cat([pc[:, :, 0], reshaped])

        res = []
        for conv in self.conv_block_0:
            res.append(conv(cat))

        for i, (r, conv) in enumerate(zip(res, self.conv_block_1)):
            res[i] = conv(r)

        for i, (r, conv) in enumerate(zip(res, self.conv_block_2)):
            res[i] = conv(r)

        res = self.reshape1(self.conv(self.cat(res)))

        return self.skip_conv(self.cat([feats, res]))

    # ----------------------------------------------------------------------
    def get_config(self):
        config = super(SEAC, self).get_config()
        config.update(
            {
                "ds": self.ds,
                "da": self.da,
                "dh": self.dh,
                "do": self.do,
                "K": self.K,
                "activation": self.activation,
                "use_bias": self.use_bias,
                "use_cache": self.use_cache,
            }
        )
        return config


# ======================================================================
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
        self.preds = preds
        self.slices = slices

    # ----------------------------------------------------------------------
    def get_graph(self, index):
        """
        Returns the i-th graph.

        Parameters
        ----------
            - index : int, index in prediction list

        Returns
        -------
            - list, of sets calohit predicted connections
        """
        return self.graphs[index]

    # ----------------------------------------------------------------------
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

    # ----------------------------------------------------------------------
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

    # ----------------------------------------------------------------------
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


# ======================================================================
class AbstractNet(Model):
    def __init__(self, name, **kwargs):
        super(AbstractNet, self).__init__(name=name, **kwargs)

    # ----------------------------------------------------------------------
    def model(self):
        pc = Input(shape=(None, 1 + self.K, self.input_dims), name="pc")
        feats = Input(shape=(None, 1 + self.K, self.f_dims), name="feats")
        return Model(
            inputs=[pc, feats], outputs=self.call([pc, feats]), name=self.name
        )

    # ----------------------------------------------------------------------
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
        preds = []
        all_slices = []
        for inp, knn_idx in zip(inputs, knn_idxs):
            N = inp[0].shape[1]

            # predict hits connections
            pred = self.predict_on_batch(inp)[0]

            # stronger prediction if both directed edges are positive
            adj = np.full([N,N], 2.)
            rows = np.repeat( np.expand_dims(np.arange(N), 1), pred.shape[1], axis=1)
            adj[rows,knn_idx[0]] = pred
            adj[adj==2.] = adj.T[adj==2.]
            adj = (adj + adj.T) / 2
            pred = np.take_along_axis(adj, knn_idx[0], axis=1)

            preds.append(pred)
            graph = [
                set(node[p > threshold])
                for node, p in zip(knn_idx[0], pred)
            ]
            graphs.append(graph)

            # DFS (depth first search)
            visited = set()  # the all visited set
            slices = []
            for node in range(len(graph)):
                if node in visited:
                    continue
                slice = set()  # the current slice only
                bfs(slice, visited, node, graph)
                slices.append(slice)

            sorted_slices = sorted(slices, key=len, reverse=True)
            all_slices.append(sorted_slices)
            state = np.zeros(N)
            for i, slice in enumerate(sorted_slices):
                state[np.array(list(slice), dtype=NP_DTYPE_INT)] = i
            status.append(state)

        return Predictions(graphs, status, preds, all_slices)
