# This file is part of SliceRL by M. Rossi
import numpy as np
from collections import namedtuple
from copy import deepcopy

EventTuple = namedtuple(
    "EventTuple",
    [
        "E",
        "x",
        "z",
        "x_dir",
        "z_dir",
        "cluster_idx",
        "pndr_idx",
        "mc_idx",
        "slicerl_idx",
    ],
)

VIEW_TO_ONEHOT_IDX = {
    "U": [1, 0, 0],
    "V": [0, 1, 0],
    "W": [0, 0, 1],
}

SWAP = np.array([[0, -1], [1, 0]])

# ======================================================================
class Event:
    """Event class wrapping different PlaneView information."""

    # ----------------------------------------------------------------------
    def __init__(self, tpc_views, min_hits=1):
        """
        Parameters
        ----------
            - tpc_views: tuple, three different tpc views arrays, each of
                         shape=(12, nb_hits)
            - min_hits : int, consider only slices with equal or more than min_hits
        """
        hits_U, hits_V, hits_W = tpc_views
        self.U = PlaneView("U", hits_U, min_hits)
        self.V = PlaneView("V", hits_V, min_hits)
        self.W = PlaneView("W", hits_W, min_hits)
        self.nb_all_clusters = (
            self.U.nb_clusters + self.V.nb_clusters + self.W.nb_clusters
        )
        # do_sanity_checks(self.U, self.V, self.W)

    # ----------------------------------------------------------------------
    def store_preds(self, slices):
        """
        Store predicted slicing info in self.status.

        Parameters
        ----------
            - slices: list, of sets containing the cluster indices that belong
                      to the same slice
        """
        predU = deepcopy(self.U.ordered_cluster_idx)
        predV = self.V.ordered_cluster_idx + self.U.nb_clusters
        predW = self.W.ordered_cluster_idx + self.U.nb_clusters + self.V.nb_clusters
        for slice in deepcopy(slices):
            idx = np.min(list(slice))
            # U plane
            mask = np.isin(predU, list(slice))
            predU[mask] = idx
            # V plane
            mask = np.isin(predV, list(slice))
            predV[mask] = idx
            # W plane
            mask = np.isin(predW, list(slice))
            predW[mask] = idx
        self.U.status = predU
        self.V.status = predV
        self.W.status = predW


# ======================================================================
class PlaneView:
    """PlaneView class keeping track of calohits in a 2D plane view."""

    # ----------------------------------------------------------------------
    def __init__(self, tpc_view, original_hits, min_hits=1):
        """
        Parameters
        ----------
            - tpc_view: str, TPC plane view
            - original_hits : np.array, shape=(8, num calohits), containing all the
                              event information. Each calohit is described by:
                              - energy [ADC/100]
                              - x coordinate [10^1 m]
                              - z coordinate [10^1 m]
                              - x expected direction
                              - z expected direction
                              - input cluster idx
                              - pandora slice index
                              - cheating slice idx (mc truth)
                              - slicing output (array of minus ones if not loading
                                inference results)
                              - test beam flag
                              - PDG
                              - pfo index
            - min_hits : int, consider only slices with equal or more than min_hits
        """
        self.tpc_view = VIEW_TO_ONEHOT_IDX[tpc_view]
        # order the calohits by x and store the idx permutation (this does
        # affect the calohit ordering directly)
        self.original_calohits = original_hits

        calohits = deepcopy(original_hits)
        idx = np.argsort(calohits[1])
        self.sorting_x_idx = np.repeat(idx.reshape([1, -1]), calohits.shape[0], axis=0)
        calohits = np.take_along_axis(calohits, self.sorting_x_idx, axis=1)

        # filter calohits belonging to cluster with size < min_hits
        if min_hits > 1:
            filter_fn = lambda x: np.count_nonzero(calohits[5] == x[5]) > min_hits
            calohits = np.stack(list(filter(filter_fn, list(calohits.T))), -1)
        self.calohits = calohits

        # check if (E,x,z) inputs are in range
        assert np.all(calohits[0] < 500)
        assert np.all(
            np.logical_and(calohits[1] >= -0.38, calohits[1] <= 0.38)
        ), f"found calohit x coordinate range: [{calohits[1].min()}, {calohits[1].max()}]"
        assert np.all(
            np.logical_and(calohits[2] >= -0.36, calohits[2] <= 0.92)
        ), f"found calohit z coordinate range: [{calohits[2].min()}, {calohits[2].max()}]"

        # build the mc slice size ordering
        # change the slice idx sorting by decreasing size (this does not affect
        # the calohit ordering)
        self.mc_idx = calohits[7]
        self.ordered_mc_idx = deepcopy(calohits[7])
        sort_fn = lambda x: np.count_nonzero(calohits[7] == x)
        self.sorted_mc_idx = sorted(list(set(calohits[7])), key=sort_fn, reverse=True)
        for i, idx in enumerate(self.sorted_mc_idx):
            self.ordered_mc_idx[calohits[7] == idx] = i

        # build the pndr_slice ordering (useful for testing)
        self.pndr_idx = calohits[6]
        self.ordered_pndr_idx = deepcopy(calohits[6])
        sort_fn = lambda x: np.count_nonzero(calohits[6] == x)
        self.sorted_pndr_idx = sorted(list(set(calohits[6])), key=sort_fn, reverse=True)
        for i, idx in enumerate(self.sorted_pndr_idx):
            self.ordered_pndr_idx[calohits[6] == idx] = i

        # build cluster list ordering
        self.cluster_idx = calohits[5]
        self.ordered_cluster_idx = deepcopy(calohits[5])
        sort_fn = lambda x: np.count_nonzero(calohits[5] == x)
        self.sorted_cluster_idx = sorted(
            list(set(calohits[5])), key=sort_fn, reverse=True
        )
        for i, idx in enumerate(self.sorted_cluster_idx):
            self.ordered_cluster_idx[calohits[5] == idx] = i

        self.cluster_set = set(self.ordered_cluster_idx)
        self.nb_clusters = len(self.cluster_set)

        # point cloud to be passed to the agent:
        # (energy, x, z, x_dir, z_dir, pfo cluster idx)
        self.point_cloud = np.concatenate([calohits[:5], [self.ordered_cluster_idx]])
        self.status = calohits[8]
        self.nb_calohits = len(self.status)

        self.test_beam = self.calohits[9]  # test beam flag array
        self.PDG = self.calohits[10]  # parent mc particle PDG array
        self.pfo_index = self.calohits[11]  # parent pfo index attribute

        (
            self.all_cluster_features,
            self.cluster_to_main_pfo,
        ) = self.get_all_cluster_info()

        # look at the do_checks function docstring
        self.calo2mpfo = self.cluster_to_main_pfo[self.ordered_cluster_idx.astype(int)]
        calo2pfo_set= set(self.calo2mpfo)
        pfo_set = set(self.pfo_index)
        self.not_visited_pfos = pfo_set.difference(calo2pfo_set)
        self.not_visited_pfos_completeness = [np.count_nonzero(self.pfo_index == pfo)/self.nb_calohits for pfo in self.not_visited_pfos]
        self.visited_pfos = pfo_set.intersection(calo2pfo_set)
        self.visited_pfos_completeness = [np.count_nonzero(self.pfo_index == pfo)/self.nb_calohits for pfo in self.visited_pfos]

    # ----------------------------------------------------------------------
    def get_all_cluster_info(self):
        """
        Compute the cluster vector information for all clusters in the plane view.
        Matches each cluster to the predominant pfo in the event.

        Returns
        -------
            - np.array, cluster features of shape=(nb_clusters, nb_features)
            - np.array, main pfo index of shape=(nb_clusters)
        """
        all_features = []
        # cluster to main pfo
        c2mpfo = []

        for idx in sorted(list(self.cluster_set)):
            # extract cluster
            m = self.ordered_cluster_idx == idx
            cluster_hits = self.point_cloud[:, m]
            hits_pct = np.count_nonzero(m) / len(self.ordered_cluster_idx)
            all_features.append(
                get_cluster_features(cluster_hits, hits_pct, self.tpc_view)
            )

            # select the main pfo for cluster
            sort_fn = lambda x: np.count_nonzero(self.pfo_index[m] == x)
            c2mpfo.append(
                sorted(list(set(self.pfo_index[m])), key=sort_fn, reverse=True)[0]
            )
        return np.stack(all_features, axis=0), np.array(c2mpfo)

    # ----------------------------------------------------------------------
    def __len__(self):
        return self.nb_calohits

    # ----------------------------------------------------------------------
    def state(self):
        """
        Return the observable state: describes the clusters adjecency matrix.

        Returns
        -------
            - np.array, of shape=(nb_clusters, nb_clusters, 2, nb_features)
        """
        return self.cluster_features

    # ----------------------------------------------------------------------
    def original_order_status(self):
        status = np.zeros_like(self.status)
        return np.put_along_axis(status, self.sorting_x_idx, self.status, axis=1)

    # ----------------------------------------------------------------------
    def store_preds(self, slices):
        """
        Store predicted slicing info in self.status.

        Parameters
        ----------
            - status : np.array, status predictions array of shape=(num calohits,)
            - graph   : np.array, graph predictions array of shape=(num calohits, num neighbors)
        """
        pred = deepcopy(self.ordered_cluster_idx)
        for slice in deepcopy(slices):
            idx = np.min(list(slice))
            slice.remove(idx)
            for cluster in slice:
                pred[pred == cluster] = idx
        self.status = pred

    # ----------------------------------------------------------------------
    def calohits_to_array(self):
        """
        Takes calohits list into a list of arrays for plot rendering.

        Returns
        -------
            - numpy array of shape (7, num calohits). Rows contain in order:
              energies, xs, zs, cluster_idx, pndr_idx, cheating_idx, slicerl_idx

        Note
        ----
            - energy is measured in ADC
            - spatial coordinates x and z are converted in cm
        """
        # remove padding and restore natural measure units
        array = deepcopy(self.original_calohits[:6])
        array[0] = array[0] * 100  # to ADC
        array[1] = array[1] * 1000  # to cm
        array[2] = array[2] * 1000  # to cm

        # concatenate pndr idx row
        array = np.concatenate(
            [
                array,  # (E, x, z, pfo cluster idx)
                [self.pndr_idx],  # size pndr idx (original order)
                [self.mc_idx],  # size mc idx   (original order)
                [self.original_order_status()],  # status vector (original order)
            ]
        )

        return array

    # ----------------------------------------------------------------------
    def calohits_to_namedtuple(self):
        """
        Takes calohits list into an EventTuple object for plot rendering.

        Returns
        -------
            - numpy array of shape (7, num calohits). Rows contain in order:
              energies, xs, zs, cluster_idx, pndr_idx, cheating_idx, slicerl_idx

        """
        arr = self.calohits_to_array()
        return EventTuple(*arr)

    # ----------------------------------------------------------------------
    def dump(self, fname):
        """ Dump calohits list to fname file """
        rows = self.calohits_to_array()
        for row in rows:
            np.savetxt(fname, row, fmt="%.5f", newline=",")
            fname.write(b"\n")


# ----------------------------------------------------------------------
def get_cluster_features(cluster_hits, hits_pct, tpc_view):
    """Compute the cluster feature vector, given a single cluster

    Parameters
    ----------
        - cluster_hits: np.array, of shape=(11, nb_cluster_hits)
        - hits_pct: float, percentage of hits contained by this cluster over
                    the total in the plane view
        - tpc_view: list, one-hot list to match U/V/W planes

    Return
    ------
        - np.array, cluster feature vector of shape=(nb_features,)
    """
    features = []

    pc = cluster_hits[1:3]
    mux, muz = pc.mean(1)

    # standardize
    std = pc.std(1, keepdims=True)
    # if there's no variance on some axis, just keep it as it is
    std[std == 0] = 1
    std_pc = (pc - pc.mean(1, keepdims=True)) / std

    # correlation matrix
    cov = np.cov(std_pc)
    # cov = np.corrcoef(std_pc) # raises
    evalues, evectors = np.linalg.eig(cov)
    pct = evalues[0] / evalues.sum()
    # eigenvectors are evector[:,i]

    # rotate the standardized cluster
    rot_pc = np.matmul(evectors, std_pc)

    # compute the cluster delimiter points
    # the box is ok along the first PCA component, not on the second one
    dists = (rot_pc[:, None] * evectors[..., None]).sum(0)
    p0, p1 = np.argmin(dists, axis=1)
    p2, p3 = np.argmax(dists, axis=1)
    delimiters = pc[:, [p0, p1, p2, p3]]

    # build the cluster feature vector
    features.append(hits_pct)  # cluster hits percentage (0)
    features.extend([mux, muz])  # cluster mean x and z (1:3)
    features.append(cluster_hits[3, 0])  # expected direction x (3)
    features.append(cluster_hits[4, 0])  # expected direction z (4)
    features.extend(evalues.flatten().tolist())  # eigenvalues (5:7)
    features.extend(evectors.flatten().tolist())  # eigenvectors (7:11)
    features.append(pct)  # eigenvalues percentage (11)
    features.extend([cov[0, 0], cov[1, 0], cov[1, 1]])  # covariance matrix (12:15)
    features.extend(delimiters.flatten().tolist())  # delimiters (15:23)
    features.append(cluster_hits[0].mean())  # cluster hits mean energy (23)
    features.append(
        cluster_hits[0].std()
    )  # cluster hits energy standard deviation (24)
    features.extend(tpc_view)
    return np.array(features)

# ======================================================================

def do_sanity_checks(U,V,W):
    """
    Performs sanity checks.

    Note
    ----
        clusters and pfos are two different calohit partitions. clusters
        operate at PlaneView level, while pfos are event-wide.
        Since 2D clustering is not a perfect algorithm, there could be
        discrepancies between the two set partitions. We must ensure that the
        discrepancies are tiny and do not spoil the slicing algorithm performance.
        Check then the pfos that are left aside when assigning the main pfo to
        a cluster, have in fact small contribution (completeness) in the
        overall event (i.e., the percentage of the calohits relative to these
        pfos is small).
    """
    import matplotlib.pyplot as plt
    from slicerl.diagnostics import norm, cmap
    bins = np.logspace(-4,0,20)
    plt.subplot(131)
    plt.title("U plane")
    plt.hist(U.visited_pfos_completeness, bins=bins, label='visited', color='green', histtype='step', lw=0.5)
    plt.hist(U.not_visited_pfos_completeness, bins=bins, label='not visited', color='red', histtype='step', lw=0.5)
    plt.legend()
    plt.xscale('log')
    plt.xlabel('Pfo completeness')
    plt.subplot(132)
    plt.title("V plane")
    plt.hist(V.visited_pfos_completeness, bins=bins, label='visited', color='green', histtype='step', lw=0.5)
    plt.hist(V.not_visited_pfos_completeness, bins=bins, label='not visited', color='red', histtype='step', lw=0.5)
    plt.xscale('log')
    plt.xlabel('Pfo completeness')
    plt.subplot(133)
    plt.title("W plane")
    plt.hist(W.visited_pfos_completeness, bins=bins, label='visited', color='green', histtype='step', lw=0.5)
    plt.hist(W.not_visited_pfos_completeness, bins=bins, label='not visited', color='red', histtype='step', lw=0.5)
    plt.xscale('log')
    plt.xlabel('Pfo completeness')
    plt.show()

    plt.subplot(331)
    plt.title("U plane")
    plt.ylabel("inputs")
    plt.scatter(U.calohits[1]*1000, U.calohits[2]*1000, s=0.5, c=U.calohits[5]%128, norm=norm, cmap=cmap)
    plt.subplot(332)
    plt.title("V plane")
    plt.scatter(V.calohits[1]*1000, V.calohits[2]*1000, s=0.5, c=V.calohits[5]%128, norm=norm, cmap=cmap)
    plt.subplot(333)
    plt.title("W plane")
    plt.scatter(W.calohits[1]*1000, W.calohits[2]*1000, s=0.5, c=W.calohits[5]%128, norm=norm, cmap=cmap)
    plt.subplot(334)
    plt.ylabel("truths")
    plt.scatter(U.calohits[1]*1000, U.calohits[2]*1000, s=0.5, c=U.calohits[7]%128, norm=norm, cmap=cmap)
    plt.subplot(335)
    plt.scatter(V.calohits[1]*1000, V.calohits[2]*1000, s=0.5, c=V.calohits[7]%128, norm=norm, cmap=cmap)
    plt.subplot(336)
    plt.scatter(W.calohits[1]*1000, W.calohits[2]*1000, s=0.5, c=W.calohits[7]%128, norm=norm, cmap=cmap)
    plt.subplot(337)
    plt.ylabel("calo to pfos")
    plt.scatter(U.calohits[1]*1000, U.calohits[2]*1000, s=0.5, c=U.calo2mpfo%128, norm=norm, cmap=cmap)
    plt.subplot(338)
    plt.scatter(V.calohits[1]*1000, V.calohits[2]*1000, s=0.5, c=V.calo2mpfo%128, norm=norm, cmap=cmap)
    plt.subplot(339)
    plt.scatter(W.calohits[1]*1000, W.calohits[2]*1000, s=0.5, c=W.calo2mpfo%128, norm=norm, cmap=cmap)

    plt.show()