# This file is part of SliceRL by M. Rossi
import logging
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from slicerl import PACKAGE
from slicerl.utils.tools import bfs

logger = logging.getLogger(PACKAGE)


class Predictions:
    """Utility class to return RandLA-Net predictions."""

    def __init__(self, y_pred, preds, slices):
        """
        Parameters
        ----------
            - y_pred : list, of predicted vectors
            - preds  : list, of predicted adj matrices
            - slices : list, of sets
        """
        self.all_y_pred = y_pred
        self.all_events_preds = preds
        self.all_events_slices = slices

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
            - np.array: status at index i of shape=(N,N)
        """
        return self.all_events_preds[index]

    # ----------------------------------------------------------------------
    def get_slices(self, index):
        """
        Returns the slices: each sl contains the cluster indices inside the
        sl set.

        Parameters
        ----------
            - index : int, index in sl list

        Returns
        -------
            - list: of set objects with calohit indices
        """
        return self.all_events_slices[index]


# ======================================================================
class AbstractNet(Model):
    """
    Network abstract class.

    The daughter class must define the `input_shape` attribute to use the model
    method.    
    """

    def __init__(self, f_dims, name, **kwargs):
        self.f_dims = f_dims
        self.inputs_shape = None
        super(AbstractNet, self).__init__(name=name, **kwargs)
        self.add_batch_dim = lambda x: (None,) + x

    # ----------------------------------------------------------------------
    def model(self):
        return Model(inputs=self.inputs_shape, outputs=self.call(self.inputs_shape), name=self.name)


# ======================================================================
def add_extension(name):
    """
    Adds extension to cumulative variables while looping over it.

    Parameters
    ----------
        - name: str, the weight name to be extended

    Returns
    -------
        - str, the extended weight name
    """
    ext = "_cum"
    l = name.split(":")
    l[-2] += ext
    return ":".join(l[:-1])


# ======================================================================
class BatchCumulativeNetwork(Model):
    """
    Network implementing gradient aggregation over mini-batch. Inheriting from
    this class overrides the Model.train_sted method, see the train_step method
    docstring.

    The daughter class must define 
    """

    def build(self, input_shape):
        """
        Builds network weights and define `tf.Variable` placeholders for
        cumulative gradients.
        """
        super(BatchCumulativeNetwork, self).build(input_shape)
        self.cumulative_gradients = [
            tf.Variable(
                tf.zeros_like(this_var),
                trainable=False,
                name=add_extension(this_var.name),
            )
            for this_var in self.trainable_variables
        ]
        self.cumulative_counter = tf.Variable(
            tf.constant(0), trainable=False, name="cum_counter"
        )

    # ----------------------------------------------------------------------
    def reset_cumulator(self):
        """
        Reset counter and gradients cumulator gradients.
        """

        for i in range(len(self.cumulative_gradients)):
            self.cumulative_gradients[i].assign(
                tf.zeros_like(self.trainable_variables[i])
            )
        self.cumulative_counter.assign(tf.constant(1))

    # ----------------------------------------------------------------------
    def increment_counter(self):
        """
        Reset counter and gradients cumulator gradients.
        """
        self.cumulative_counter.assign_add(tf.constant(1))

    # ----------------------------------------------------------------------
    def train_step(self, data):
        """
        The network accumulates the gradients according to batch size, to allow
        gradient averaging over multiple inputs. The aim is to reduce the loss
        function fluctuations.
        """
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        if self.verbose:
            reset_2 = (self.cumulative_counter + 1) % self.batch_size == 0
            tf.cond(
                reset_2,
                lambda: print_loss(x, y, y_pred, loss),
                lambda: False,
            )

        reset = self.cumulative_counter % self.batch_size == 0
        tf.cond(reset, self.reset_cumulator, self.increment_counter)

        # Compute gradients
        trainable_vars = self.trainable_variables

        gradients = tape.gradient(loss, trainable_vars)
        for i, grad in enumerate(gradients):
            self.cumulative_gradients[i].assign_add(grad / self.batch_size)

        # Update weights
        reset = self.cumulative_counter % self.batch_size == 0

        if self.verbose:
            reset_1 = (self.cumulative_counter - 1) % self.batch_size == 0
            tf.cond(
                reset_1,
                lambda: print_gradients(zip(self.cumulative_gradients, trainable_vars)),
                lambda: False,
            )

        tf.cond(
            reset,
            lambda: self.optimizer.apply_gradients(
                zip(self.cumulative_gradients, trainable_vars)
            ),
            lambda: False,
        )

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


# ======================================================================
def print_loss(x, y, y_pred, loss):
    # tf.print(
    #     ", x:",
    #     tf.reduce_mean(x),
    #     tf.math.reduce_std(x),
    #     ", y:",
    #     y,
    #     ", y_pred:",
    #     y_pred,
    #     ", loss:",
    #     loss,
    # )
    tf.print(y, y_pred)
    return True


# ======================================================================
def print_gradients(gradients):
    for g, t in gradients:
        tf.print(
            "Param:",
            g.name,
            ", value:",
            tf.reduce_mean(t),
            tf.math.reduce_std(t),
            ", grad:",
            tf.reduce_mean(g),
            tf.math.reduce_std(g),
        )
    tf.print("---------------------------")
    return True


# ======================================================================
def get_prediction(net, test_generator, batch_size, threshold=0.5):
    """
    Predict over a iterable of inputs

    Parameters
    ----------
        - net: AbstractNet, network to get predictions
        - test_generator: EventDataset, generator for inference
        - batch_size: int
        - threshold: float, interpret as positive if above network
                     prediction is threshold

    Returns
    -------
        Predictions object
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
            net.predict(ii[None], batch_size, verbose=0).flatten() for ii in tqdm(inp)
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
    return Predictions(y_pred, preds, all_slices)


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
