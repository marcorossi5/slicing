# This file is part of SliceRL by M. Rossi
import numpy as np
from collections import namedtuple
from copy import deepcopy

EventTuple = namedtuple("EventTuple", ["E", "x", "z", "cluster_idx", "pndr_idx",
                                       "mc_idx", "slicerl_idx"])

#======================================================================
class Event:
    """Event class keeping track of calohits in a 2D plane view."""

    #----------------------------------------------------------------------
    def __init__(self, original_hits, min_hits=1, max_hits=15000):
        """
        Parameters
        ----------
            - original_hits : np.array, shape=(7, num calohits), containing all the
                              event information. Each calohit is described by:
                              - energy [ADC/100]
                              - x coordinate [10^1 m]
                              - z coordinate [10^1 m]
                              - input pfo list cluster idx
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
        self.sorting_x_idx = np.repeat(idx.reshape([1,-1]), calohits.shape[0], axis=0)
        calohits = np.take_along_axis(calohits, self.sorting_x_idx, axis=1)

        if min_hits > 1:
            filter_fn = lambda x: np.count_nonzero(calohits[5] == x[5]) > min_hits
            calohits = np.stack(list(filter(filter_fn, list(calohits.T))), -1)
        self.calohits = calohits

        # check if (E,x,z) inputs are in range
        assert np.all(calohits[0] < 500)
        assert np.all(np.logical_and(calohits[1] >= -0.38,  calohits[1] <= 0.38)), \
            f"found calohit x coordinate range: [{calohits[1].min()}, {calohits[1].max()}]"
        assert np.all(np.logical_and(calohits[2] >= -0.36, calohits[2] <= 0.92)), \
            f"found calohit z coordinate range: [{calohits[2].min()}, {calohits[2].max()}]"

        # build the mc slice size ordering
        # TODO: mc_idx contains some -1, think about masking out those
        # un-associated calohits and then make the ordering
        self.mc_idx         = calohits[5]
        self.ordered_mc_idx = deepcopy(calohits[5])
        sort_fn = lambda x: np.count_nonzero(calohits[5] == x)
        self.sorted_mc_idx = sorted(list(set(calohits[5])), key=sort_fn, reverse=True)
        for i, idx in enumerate(self.sorted_mc_idx):
            self.ordered_mc_idx[calohits[5] == idx] = i

        # build the pndr_slice ordering (useful for testing)
        self.pndr_idx         = calohits[4]
        self.ordered_pndr_idx = deepcopy(calohits[4])
        sort_fn = lambda x: np.count_nonzero(calohits[4] == x)
        self.sorted_pndr_idx = sorted(list(set(calohits[4])), key=sort_fn, reverse=True)
        for i, idx in enumerate(self.sorted_pndr_idx):
            self.ordered_pndr_idx[calohits[4] == idx] = i

        # build pfo cluster list ordering
        self.cluster_idx         = calohits[3]
        self.ordered_cluster_idx = deepcopy(calohits[3])
        sort_fn = lambda x: np.count_nonzero(calohits[3] == x)
        self.sorted_cluster_idx = sorted(list(set(calohits[3])), key=sort_fn, reverse=True)
        for i, idx in enumerate(self.sorted_cluster_idx):
            self.ordered_cluster_idx[calohits[3] == idx] = i

        # point cloud to be passed to the agent:
        # (energy, x, z, pfo cluster idx)
        self.point_cloud  = np.concatenate([calohits[:3], [self.ordered_cluster_idx]])
        self.status = calohits[6]
        self.num_calohits = len(self.status)

    #----------------------------------------------------------------------
    def __len__(self):
        return self.num_calohits

    #----------------------------------------------------------------------
    def state(self, step=None, is_training=False):
        """
        Return the observable state: padded point cloud, describing
        (E, x, z, pfo cluster idx, current status) for each calohit

        Parameter
        ---------
            - step        : int, episode_step (needed during training only)
            - is_training : bool, training mode or not

        Returns
        -------
            - array of shape=(max_hits, 4)
        """
        return self.point_cloud.T

    #----------------------------------------------------------------------
    def original_order_status(self):
        status = np.zeros_like(self.status)
        return np.put_along_axis(status, self.sorting_x_idx, self.status, axis=1)
    #----------------------------------------------------------------------
    def store_preds(self, pred):
        """
        Store predicted slicing info in self.status.

        Parameters
        ----------
            - pred : np.array, predictions array of shape=(num calohits,)
        """
        self.status = pred

    #----------------------------------------------------------------------
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
        array    = deepcopy(self.original_calohits[:4])
        array[0] = array[0] * 100  # to ADC
        array[1] = array[1] * 1000 # to cm
        array[2] = array[2] * 1000 # to cm

        # concatenate pndr idx row
        array = np.concatenate([
                        array,                   # (E, x, z, pfo cluster idx)
                        [self.pndr_idx],         # size pndr idx (original order)
                        [self.mc_idx],           # size mc idx   (original order)
                        [self.original_order_status()] # status vector (original order)
                    ])

        return array

    #----------------------------------------------------------------------
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

    #----------------------------------------------------------------------
    def dump(self, fname):
        """ Dump calohits list to fname file """
        rows = self.calohits_to_array()
        for row in rows:
            np.savetxt(fname, row, fmt='%.5f', newline=',')
            fname.write(b"\n")

# TODO: think about normalizing the cluster_idx and status inputs to the actor model