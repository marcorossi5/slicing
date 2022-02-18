# This file is part of SliceRL by M. Rossi
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.activations import sigmoid
from .AbstractNet import AbstractNet, BatchCumulativeNetwork
from .FFNN import Head
from .layers import TransformerEncoder


class CMNet(AbstractNet, BatchCumulativeNetwork):
    """Class deifining CM-Net: Cluster merging network."""

    def __init__(
        self,
        batch_size=1,
        f_dims=2,
        fc_heads=5,
        mha_heads=5,
        activation="relu",
        use_bias=True,
        verbose=False,
        name="CM-Net",
        **kwargs,
    ):
        """
        Parameters
        ----------
            - f_dims     : int, point cloud feature dimensions
            - activation : str, default layer activation
            - use_bias   : bool, wether to use bias or not
        """
        super(CMNet, self).__init__(f_dims=f_dims, name=name, **kwargs)

        # store args
        self.batch_size = int(batch_size)
        self.f_dims = f_dims
        self.fc_heads = fc_heads
        self.mha_heads = mha_heads
        self.activation = activation
        self.use_bias = use_bias
        self.verbose = verbose

        self.mha_filters = [64, 128]
        self.mhas = []
        # self.mha_ifilters = [self.fc_heads] + self.mha_filters[:-1]
        for ih, dout in enumerate(self.mha_filters):
            # self.mhas.append(MyGRU(dout, self.mha_heads, name=f"mha_{ih}"))
            self.mhas.append(TransformerEncoder(dout, self.mha_heads, name=f"mha_{ih}"))

        # self.cumulative_gradients = None
        self.cumulative_counter = tf.constant(0)

        # store some useful parameters for the fc layers
        self.fc_filters = [64, 32, 16, 8, 4, 2, 1]
        # self.dropout_idxs = [0, 1]
        self.dropout_idxs = []
        self.dropout = 0.2
        self.heads = []
        for ih in range(self.fc_heads):
            self.heads.append(
                Head(
                    self.fc_filters,
                    ih,
                    self.dropout_idxs,
                    self.dropout,
                    activation=self.activation,
                    name=f"head_{ih}",
                )
            )
        self.concat = Concatenate(axis=-1, name="cat")

        self.inputs_shape = Input(shape=(None, self.f_dims), name="clusters_hits")

        # explicitly build network weights
        super(CMNet, self).build(self.add_batch_dim(self.inputs_shape))

    # ----------------------------------------------------------------------
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
        x = inputs
        # x = tf.expand_dims(inputs, 0)
        for mha in self.mhas:
            x = self.activation(mha(x))

        x = tf.reduce_max(x, axis=1)

        results = []
        for head in self.heads:
            results.append(head(x))

        return tf.reduce_mean(sigmoid(self.concat(results)), axis=-1)    
