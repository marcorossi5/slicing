# This file is part of SliceRL by M. Rossi
import logging
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.activations import relu, softmax
from slicerl import PACKAGE
from .AbstractNet import BatchCumulativeNetwork
from .layers import TransformerEncoder, Head

logger = logging.getLogger(PACKAGE + ".HCNet")


class HCNet(BatchCumulativeNetwork):
    """
    Class deifining HC-Net: Hitlevel Clustering Network.

    In this approach the network is trained to predict the slice index at the
    hit level: input is a point cloud in 2D with additional features per hit
    of shape (1,[nb hits],nb feats), output is a tensor of shape
    ([nb hits],units) containing probabilities that an hit belongs to a specific
    slice. The indices of the slices are numbered with decreasingly in size.

    Note: since the number of hits may vary depending on the cluster pairs,
    multiple forward passes cannot be parallelized in a batch. Training
    gradients can nonetheless be avaraged over multiple examples.
    """

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
        units=128,
        f_dims=2,
        nb_mha_heads=1,
        mha_filters=[16, 32, 64],
        kernel_transformation="softmax",
        projection_matrix_type="random",
        nb_random_features=128,
        nb_fc_heads=1,
        fc_filters=[64],
        batch_size=1,
        activation=relu,
        use_bias=True,
        verbose=False,
        name="HC-Net",
        **kwargs,
    ):
        """
        Parameters
        ----------
            - units: int, the number of output classes
            - f_dims: int, number of point cloud feature dimensions
            - nb_mha_heads: int, the number of heads in the `MultiHeadAttention` layer
            - mha_filters: list, the output units for each `MultiHeadAttention` in the stack
            - kernel_transformation: str, available options softmax | relu
            - projection_matrix_type: str, available options random | None
            - nb_random_features: int, favor+ random features number
            - nb_fc_heads: int, the number of `Head` layers to be concatenated
            - fc_filters: list, the output units for each `Head` in the stack
            - batch_size: int, the effective batch size for gradient descent
            - activation: str, default keras layer activation
            - use_bias: bool, wether to use bias or not
            - verbose: str, wether to print extra training information
            - name: str, the name of the neural network instance
        """
        super(HCNet, self).__init__(name=name, **kwargs)

        # store args
        self.units = units
        self.f_dims = f_dims
        self.nb_mha_heads = nb_mha_heads
        self.mha_filters = mha_filters
        self.kernel_transformation = kernel_transformation
        self.projection_matrix_type = projection_matrix_type
        self.nb_random_features = nb_random_features
        self.nb_fc_heads = nb_fc_heads
        self.fc_filters = fc_filters
        self.batch_size = int(batch_size)
        self.activation = activation
        self.use_bias = use_bias
        self.verbose = verbose

        # adapt the output if requested
        if self.units != self.fc_filters[-1]:
            logger.warning(
                f"Units (found {self.units}) are not compatible with requested "
                f"number of neurons in last layer (found {self.fc_filters[-1]}): "
                f"adapting last layer ..."
            )
            self.fc_filters.append(self.units)

        # attention layers
        self.mhas = [
            TransformerEncoder(
                dout,
                self.nb_mha_heads,
                attention_type="favor+",
                kernel_transformation=self.kernel_transformation,
                projection_matrix_type=self.projection_matrix_type,
                nb_random_features=self.nb_random_features,
                name=f"mha_{ih}",
            )
            for ih, dout in enumerate(self.mha_filters)
        ]

        # feed-forward layers
        self.heads = [
            Head(
                self.fc_filters,
                activation=self.activation,
                # kernel_initializer="GlorotNormal",
                name=f"head_{ih}",
            )
            for ih in range(self.nb_fc_heads)
        ]

        # explicitly build network weights
        self.inputs_shape = (None, self.f_dims)
        self.input_layer = Input(shape=self.inputs_shape, name="clusters_hits")
        super(HCNet, self).build((1,) + self.inputs_shape)

    # ----------------------------------------------------------------------
    def call(self, inputs):
        """
        Parameters
        ----------
            - inputs: tf.Tensor, point cloud of hits of shape=(1,[nb hits],f_dims)

        Returns
        -------
            tf.Tensor, hit class prbabilities of shape=(1,N,units)
        """
        x = inputs
        for mha in self.mhas:
            x = self.activation(mha(x))

        results = []
        for head in self.heads:
            results.append(head(x))

        stack = tf.stack(results, axis=-1)
        reduced = tf.reduce_mean(stack, axis=-1)
        return softmax(reduced)


# ======================================================================
# HC-Net inference
def inference(network, test_generator):
    """
    HC-Net prediction over a iterable of inputs

    Parameters
    ----------
        - network: AbstractNet, network to get predictions
        - test_generator: EventDataset, generator for inference

    Returns
    -------
        - list, of np.array predictions, each of shape=(nb hits, nb feats)
    """
    inputs = test_generator.inputs
    preds = [network.predict(ii[None], verbose=0)[0] for ii in tqdm(inputs)]
    preds = np.array(preds, dtype=object)
    return preds
