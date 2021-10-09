from slicerl.tools import bfs
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input


# ======================================================================
class Predictions:
    """ Utility class to return RandLA-Net predictions. """

    def __init__(self, preds, slices):
        """
        Parameters
        ----------
            - preds  : list, each element is a np.array with shape=(N,N)
            - slices : list, of sets
        """
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
        Returns the slices: each slice contains the cluster indices inside the
        slice set.

        Parameters
        ----------
            - index : int, index in slice list

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
    def get_prediction(self, inputs, batch_size, threshold=0.5):
        """
        Predict over a iterable of inputs

        Parameters
        ----------
            - inputs: list, events clusters feature vectors, each element has
                      shape (nb_cluster_pairs, f_dims)
            - batch_size: int
            - threshold: float, interpret as positive if above network
                         prediction is threshold

        Returns
        -------
            Predictions object
        """
        preds = []
        all_slices = []
        for inp in inputs:
            # predict cluster pair connections
            pred = self.predict(inp, batch_size, verbose=1).flatten()

            nb_clusters = (1 + np.sqrt(1 + 8 * len(pred))) / 2
            assert nb_clusters.is_integer()
            nb_clusters = int(nb_clusters)

            adj = np.zeros([nb_clusters, nb_clusters])
            k = 0
            for i in range(nb_clusters):
                for j in range(i):
                    if i == j:
                        continue
                    adj[i, j] = pred[k]
                    k += 1

            adj += adj.T + np.eye(nb_clusters)
            preds.append(adj)
            # threshold to find positive and negative edges
            pred = pred > threshold

            graph = [set(np.argwhere(merge)[:, 0]) for merge in adj]

            # BFS (breadth first search)
            visited = set()  # the all visited set
            slices = []
            for node in range(len(graph)):
                if node in visited:
                    continue
                slice = set()  # the current slice only
                bfs(slice, visited, node, graph)
                slices.append(slice)

            all_slices.append(slices)
        return Predictions(preds, all_slices)
