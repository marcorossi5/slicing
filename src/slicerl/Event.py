# This file is part of SliceRL by M. Rossi
import numpy as np
from collections import namedtuple
from copy import deepcopy

from numpy.linalg.linalg import LinAlgError, eig

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

SWAP = np.array([[0, -1], [1, 0]])

# ======================================================================
class Event:
    """Event class keeping track of calohits in a 2D plane view."""

    # ----------------------------------------------------------------------
    def __init__(self, original_hits, min_hits=1, max_hits=15000):
        """
        Parameters
        ----------
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
            - min_hits : int, consider only slices with equal or more than min_hits
            - max_hits : int, max hits to be processed by network
        """
        # order the calohits by x and store the idx permutation
        self.original_calohits = original_hits

        calohits = deepcopy(original_hits)
        idx = np.argsort(calohits[1])
        self.sorting_x_idx = np.repeat(
            idx.reshape([1, -1]), calohits.shape[0], axis=0
        )
        calohits = np.take_along_axis(calohits, self.sorting_x_idx, axis=1)

        if min_hits > 1:
            filter_fn = (
                lambda x: np.count_nonzero(calohits[5] == x[5]) > min_hits
            )
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
        self.mc_idx = calohits[7]
        self.ordered_mc_idx = deepcopy(calohits[7])
        sort_fn = lambda x: np.count_nonzero(calohits[7] == x)
        self.sorted_mc_idx = sorted(
            list(set(calohits[7])), key=sort_fn, reverse=True
        )
        for i, idx in enumerate(self.sorted_mc_idx):
            self.ordered_mc_idx[calohits[7] == idx] = i

        # build the pndr_slice ordering (useful for testing)
        self.pndr_idx = calohits[6]
        self.ordered_pndr_idx = deepcopy(calohits[6])
        sort_fn = lambda x: np.count_nonzero(calohits[6] == x)
        self.sorted_pndr_idx = sorted(
            list(set(calohits[6])), key=sort_fn, reverse=True
        )
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
        self.point_cloud = np.concatenate(
            [calohits[:5], [self.ordered_cluster_idx]]
        )
        self.status = calohits[8]
        self.num_calohits = len(self.status)

        self.cluster_features = self.get_cluster_features()
        self.target_adj = self.get_cluster_target()

        """
        import matplotlib.pyplot as plt
        from slicerl.diagnostics import cmap, norm
        from slicerl.tools import bfs

        pred = deepcopy(self.ordered_cluster_idx)

        graph = [
                set(np.argwhere(merge==1)[:,0])
                for merge in adj
            ]
        # DFS (depth first search)
        visited = set()  # the all visited set
        slices = []
        for node in range(len(graph)):
            if node in visited:
                continue
            slice = set()  # the current slice only
            bfs(slice, visited, node, graph)
            slices.append(slice)
        
        for slice in deepcopy(slices):
            idx = np.min(list(slice))
            slice.remove(idx)
            for cluster in slice:
                pred[pred == cluster] = idx
        
        print("Initial clusters: ", len(set(self.ordered_cluster_idx)))
        print("Predicted slices: ", len(set(pred)))
        print("mc slices: ", len(set(self.ordered_mc_idx)))
        print("-------------------------")
        
        fig = plt.figure()
        ax0 = fig.add_subplot(131)
        ax0.scatter(self.point_cloud[1], self.point_cloud[2], s=3, cmap=cmap, norm=norm, c=self.ordered_cluster_idx%128)
        
        ax1 = fig.add_subplot(132)
        ax1.scatter(self.point_cloud[1], self.point_cloud[2], s=3, cmap=cmap, norm=norm, c=pred%128)

        ax2 = fig.add_subplot(133)
        ax2.scatter(self.point_cloud[1], self.point_cloud[2], s=3, cmap=cmap, norm=norm, c=self.ordered_mc_idx%128)

        #for x, z, i1, p, i2 in zip(self.point_cloud[1], self.point_cloud[2], self.ordered_cluster_idx, pred, self.ordered_mc_idx):
        #    ax0.annotate(f"{int(i1):d}", (x,z), fontsize=3)
        #    # ax1.annotate(f"{int(p):d}", (x,z), fontsize=3)
        #    # ax2.annotate(f"{int(i2):d}", (x,z), fontsize=3)


        plt.show()        

        # sorted_slices = sorted(slices, key=len, reverse=True)
        # all_slices.append(sorted_slices)
        # state = np.zeros(N)
        # for i, slice in enumerate(sorted_slices):
        #     state[np.array(list(slice), dtype=NP_DTYPE_INT)] = i
        # status.append(state)
        """

    # ----------------------------------------------------------------------
    def get_cluster_features(self):
        """
        Compute the adjecency matrix for the all clusters features.

        Returns
        -------
            - np.array, of shape=(nb_clusters, nb_clusters, 2, nb_features)
        """
        all_features = []
        for idx in self.cluster_set:
            features = []
            mask = self.ordered_cluster_idx == idx

            pc = self.point_cloud[1:3, mask]
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

            features.append(
                np.count_nonzero(mask) / len(self.ordered_cluster_idx)
            )  # cluster hits percentage
            features.extend([mux, muz])  # cluster mean x and z
            features.append(
                self.point_cloud[3, mask][0]
            )  # expected direction x
            features.append(
                self.point_cloud[4, mask][0]
            )  # expected direction z
            features.extend(evalues.flatten().tolist())  # eigenvalues
            features.extend(evectors.flatten().tolist())  # eigenvectors
            features.append(pct)  # eigenvalues percentage
            features.extend(
                [cov[0, 0], cov[1, 0], cov[1, 1]]
            )  # covariance matrix
            features.extend(delimiters.flatten().tolist())  # delimiters
            features.append(
                self.point_cloud[0, mask].mean()
            )  # cluster hits mean energy
            features.append(
                self.point_cloud[0, mask].std()
            )  # cluster hits energy standard deviation

            all_features.append(np.array(features))
        all_features = np.stack(all_features, axis=0)
        rows = np.repeat(all_features[:, None], self.nb_clusters, axis=1)
        cols = np.repeat(all_features[None], self.nb_clusters, axis=0)
        return np.stack([rows, cols], axis=-2)

    # ----------------------------------------------------------------------
    def __len__(self):
        return self.num_calohits

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
    def get_cluster_target(self):
        """
        Return the perfect target state for supervised learning. Describes the
        clusters adjecency matrix.

        Returns
        -------
            - array of shape=(nb_clusters, nb_clusters)
        """
        adj = np.eye(self.nb_clusters)
        for i in range(self.nb_clusters):
            for j in range(i):
                mask_0 = self.ordered_cluster_idx == i
                mask_1 = self.ordered_cluster_idx == j
                mask = np.logical_or(mask_0, mask_1)
                mc_masked = self.ordered_mc_idx[mask]
                mc_set = set(mc_masked)

                if len(mc_set) == 1:
                    adj[i, j] = 1
                else:
                    pct = [
                        np.count_nonzero(mc_masked == idx) for idx in mc_set
                    ]
                    pct.sort(reverse=True)
                    pct = np.array(pct)

                    if pct[1] < 5 and pct[0] / pct.sum() > 0.9:
                        adj[i, j] = 1
                    else:
                        adj[i, j] = 0

        return adj + adj.T - np.eye(self.nb_clusters)

    # ----------------------------------------------------------------------
    def original_order_status(self):
        status = np.zeros_like(self.status)
        return np.put_along_axis(
            status, self.sorting_x_idx, self.status, axis=1
        )

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
                [
                    self.original_order_status()
                ],  # status vector (original order)
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


# TODO: think about normalizing the cluster_idx and status inputs to the actor model
