# This file is part of SliceRL by M. Rossi
import logging
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.activations import relu, sigmoid
from slicerl import PACKAGE
from .AbstractNet import AbstractNet, BatchCumulativeNetwork
from .layers import TransformerEncoder, Head

logger = logging.getLogger(PACKAGE + ".CMNet")


class CMNet(AbstractNet, BatchCumulativeNetwork):
    """
    Class deifining CM-Net: Cluster merging network.

    In this approach the network is trained as a binary classifier. Input is
    the union of points coming from a pair of 2D clusters accompained with a
    fixed size vector of features for each hit, of shape
    ([nb hits in clusters],nb feats). Output is the probability that the two
    clusters belong to the same slice.

    Note: since the number of hits may vary depending on the cluster pairs,
    multiple forward passes cannot be parallelized in a batch. Training
    gradients can nonetheless be avaraged over multiple examples.
    """

    def __init__(
        self,
        f_dims=2,
        nb_mha_heads=5,
        mha_filters=[64, 128],
        nb_fc_heads=5,
        fc_filters=[64, 32, 16, 8, 4, 2, 1],
        batch_size=1,
        activation=relu,
        use_bias=True,
        verbose=False,
        name="CM-Net",
        **kwargs,
    ):
        """
        Parameters
        ----------
            - f_dims: int, number of point cloud feature dimensions
            - nb_mha_heads: int, the number of heads in the `MultiHeadAttention` layer
            - mha_filters: list, the output units for each `MultiHeadAttention` in the stack
            - nb_fc_heads: int, the number of `Head` layers to be concatenated
            - fc_filters: list, the output units for each `Head` in the stack
            - batch_size: int, the effective batch size for gradient descent
            - activation: str, default keras layer activation
            - use_bias: bool, wether to use bias or not
            - verbose: str, wether to print extra training information
            - name: str, the name of the neural network instance
        """
        super(CMNet, self).__init__(name=name, **kwargs)

        # store args
        self.f_dims = f_dims
        self.nb_mha_heads = nb_mha_heads
        self.mha_filters = mha_filters
        self.nb_fc_heads = nb_fc_heads
        self.fc_filters = fc_filters
        self.batch_size = int(batch_size)
        self.activation = activation
        self.use_bias = use_bias
        self.verbose = verbose

        # adapt the output if requested
        if self.fc_filters[-1] != 1:
            logger.warning(
                f"CM-Net last layer must have one neuron only, but found "
                f"{self.fc_filters[-1]}: adapting last layer ..."
            )
            self.fc_filters.append(self.units)

        # attention layers
        self.mhas = [
            TransformerEncoder(dout, self.nb_mha_heads, attention_type="original", name=f"mha_{ih}")
            for ih, dout in enumerate(self.mha_filters)
        ]

        # feed-forward layers
        self.heads = [
            Head(
                self.fc_filters,
                activation=self.activation,
                name=f"head_{ih}",
            )
            for ih in range(self.nb_fc_heads)
        ]
        self.concat = Concatenate(axis=-1, name="cat")

        # explicitly build network weights
        build_with_shape = (None, self.f_dims)
        self.input_layer = Input(shape=build_with_shape, name="clusters_hits")
        super(CMNet, self).build((1,) + build_with_shape)

    # ----------------------------------------------------------------------
    def call(self, x):
        """
        Parameters
        ----------
            - x: tf.Tensor, point cloud of hits of shape=(1,[nb hits],f_dims)

        Returns
        -------
            tf.Tensor, merging probability of shape=(1,)
        """
        for mha in self.mhas:
            x = self.activation(mha(x))

        x = tf.reduce_max(x, axis=1)

        results = []
        for head in self.heads:
            results.append(head(x))

        return tf.reduce_mean(sigmoid(self.concat(results)), axis=-1)
