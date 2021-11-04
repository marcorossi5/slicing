from slicerl.tools import bfs
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input


# ======================================================================
class Predictions:
    """ Utility class to return RandLA-Net predictions. """

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
    def __init__(self, f_dims, name, **kwargs):
        self.f_dims = f_dims
        super(AbstractNet, self).__init__(name=name, **kwargs)

    # ----------------------------------------------------------------------
    def model(self):
        inputs = Input(shape=(None, None, 2, self.f_dims), name="pc")
        return Model(inputs=inputs, outputs=self.call(inputs), name=self.name)

    # ----------------------------------------------------------------------
    def get_prediction(self, test_generator, batch_size, threshold=0.5):
        """
        Predict over a iterable of inputs

        Parameters
        ----------
            - test_generator: EventDataset, generator for inference
            - batch_size: int
            - threshold: float, interpret as positive if above network
                         prediction is threshold

        Returns
        -------
            Predictions object
        """
        inputs = test_generator.bal_inputs
        nb_clusters_list = test_generator.nb_clusters_list
        y_pred = []
        preds = []
        all_slices = []
        if nb_clusters_list is None:
            raise ValueError(
                "Total number of cluster is unknown, did you forget to pass a valid generator for inference?"
            )
        for inp, nb_planes_clusters, cthreshold in zip(
            inputs, nb_clusters_list, test_generator.cthresholds
        ):
            # predict cluster pair connections
            out = []
            from tqdm import tqdm
            for i in tqdm(inp):
                ii = np.expand_dims(i,0)
                out.append(self.predict(ii, batch_size, verbose=0).flatten())
            pred = np.concatenate(out)
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
    print(f"Number of clusters: {nb_clusters}")
    adj_mod = np.concatenate([np.arange(nb_clusters).reshape(1, -1), adj], axis=0)
    extra_line = np.concatenate([[-1], np.arange(nb_clusters)])
    adj_mod = np.concatenate([extra_line.reshape(-1, 1), adj_mod], axis=1)

    lw = 400
    np.set_printoptions(precision=2, suppress=True, threshold=sys.maxsize, linewidth=lw)
    # print(adj_mod)
    print(adj)

    pred = (adj > 0.5).astype(int)
    print(pred)
    edges = np.argwhere(pred)
    m = edges[:, 0] > edges[:, 1]
    # print(edges[m])

    for sl in slices:
        print(sl)


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
