# This file is part of SliceRL by M. Rossi
import logging
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.activations import relu, sigmoid
from slicerl import PACKAGE
from slicerl.utils.tools import bfs
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
            TransformerEncoder(
                dout, self.nb_mha_heads, attention_type="original", name=f"mha_{ih}"
            )
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


# ======================================================================
# CM-Net inference
def inference(network, test_generator, batch_size, threshold=0.5):
    """
    CM-Net prediction over a iterable of inputs

    Parameters
    ----------
        - network: AbstractNet, network to get predictions
        - test_generator: EventDataset, generator for inference
        - batch_size: int
        - threshold: float, interpret as positive if network prediction is
                     above threshold

    Returns
    -------
        - np.array, network predictions of shape=()
        - list, of np.arrays graph adjacency matrices, each of shape=()
        - list, of sets representing the slices
    """
    inputs = test_generator.inputs
    test_cthresholds = test_generator.cthresholds
    nb_clusters_list = test_generator.nb_clusters_list
    y_pred = []
    preds = []
    all_slices = []
    if nb_clusters_list is None:
        raise ValueError(
            "Total number of cluster is unknown, did you forget to pass a valid generator for inference?"
        )
    # zipped = tqdm(zip(inputs, nb_clusters_list, test_cthresholds))
    zipped = zip(inputs, nb_clusters_list, test_cthresholds)
    for inp, nb_planes_clusters, cthreshold in zipped:
        # predict cluster pair connections
        pred = [
            network.predict(ii[None], batch_size, verbose=0).flatten()
            for ii in tqdm(inp)
        ]
        pred = np.concatenate(pred)
        # pred = np.load("../output/test/pred.npy")
        np.save("../output/test/pred.npy", pred)
        # pred = net.predict(inp, batch_size, verbose=0).flatten()

        y_pred.append(pred)
        # nb_clusters = (1 + np.sqrt(1 + 8 * len(pred))) / 2
        nb_clusters = sum(nb_planes_clusters)
        # assert nb_clusters.is_integer()
        # nb_clusters = int(nb_clusters)
        adj = np.zeros([nb_clusters, nb_clusters])
        # build a block diagonal matrix
        k = 0
        ped = 0
        import sys

        np.set_printoptions(linewidth=700, precision=1, threshold=sys.maxsize)
        for nb_plane_clusters, cts in zip(nb_planes_clusters, cthreshold):
            for i in range(nb_plane_clusters):
                for j in range(i):
                    if i == j:
                        continue
                    adj[ped + i, ped + j] = pred[k]
                    k += 1
            # prevent large cluster mixing due to small clusters connections
            clear_plane_large_small_interactions(adj, ped, nb_plane_clusters, cts)
            clear_plane_small_small_interactions(adj, ped, nb_plane_clusters, cts)
            ped += nb_plane_clusters
        adj += adj.T + np.eye(nb_clusters)
        preds.append(adj)
        # threshold to find positive and negative edges
        pred = adj > threshold
        graph = [set(np.argwhere(merge)[:, 0]) for merge in pred]
        # BFS (breadth first search)
        visited = set()  # the all visited set
        slices = []
        for node in range(len(graph)):
            if node in visited:
                continue
            sl = set()  # the current sl only
            bfs(sl, visited, node, graph)
            slices.append(sl)
        all_slices.append(slices)
        # print_prediction(adj, all_slices)
    return y_pred, preds, all_slices


# ======================================================================
def print_prediction(adj, slices):
    """Prints to std output the adj matrix and all the slices."""
    import sys

    nb_clusters = len(adj)
    logger.debug(f"Number of clusters: {nb_clusters}")
    adj_mod = np.concatenate([np.arange(nb_clusters).reshape(1, -1), adj], axis=0)
    extra_line = np.concatenate([[-1], np.arange(nb_clusters)])
    adj_mod = np.concatenate([extra_line.reshape(-1, 1), adj_mod], axis=1)

    lw = 400
    np.set_printoptions(precision=2, suppress=True, threshold=sys.maxsize, linewidth=lw)
    # logger.debug(adj_mod)
    logger.debug(adj)

    pred = (adj > 0.5).astype(int)
    logger.debug(pred)
    # edges = np.argwhere(pred)
    # m = edges[:, 0] > edges[:, 1]
    # logger.debug(edges[m])

    for sl in slices:
        logger.debug(sl)


# ======================================================================
def clear_plane_large_small_interactions(adj, ped, nb_clusters, cts):
    """
    Modifies the adjecency matrix to allow at maximum one interaction with large
    clusters for small size ones. This prevents clusters that has low size to
    cause merging of higher order clusters. This is done on a plane view basis.
    In place operation.

    Parameters
    ----------
        - adj: np.array, adjecency matrix of shape=(nb_all_clusters, nb_all_clusters)
        - ped: int, start index of the plane submatrix
        - nb_clusters: int, number of clusters in plane view
        - cts: list, starting indices of the different cluster size levels
    """
    bins = list(ped + np.array(cts + [nb_clusters]))
    for ct1, ct2 in zip(bins, bins[1:]):
        block = np.s_[ct1:ct2, ped:ct1]
        m = np.amax(adj[block], axis=-1, keepdims=True)
        mask = (adj[block] == m).astype(float)
        adj[block] = mask * m


# ======================================================================
def clear_plane_small_small_interactions(adj, ped, nb_clusters, cts):
    """
    Modifies the adjecency matrix to disallow small to small clusters
    interactions. As a result, one small cluster can be linked to a big cluster
    only. This is done on a plane view basis. In place operation.

    Parameters
    ----------
        - adj: np.array, adjecency matrix of shape=(nb_all_clusters, nb_all_clusters)
        - ped: int, start index of the plane submatrix
        - nb_clusters: int, number of clusters in plane view
        - cts: list, starting indices of the different cluster size levels
    """
    istart = ped + cts[-1]
    iend = ped + nb_clusters
    nb_small_clusters = nb_clusters - cts[-1]
    block = np.s_[istart:iend, istart:iend]
    adj[block] = np.zeros([nb_small_clusters, nb_small_clusters])
