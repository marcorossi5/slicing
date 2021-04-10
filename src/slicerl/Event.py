# This file is part of SliceRL by M. Rossi

import sys
import numpy as np
import math
import json, gzip
from collections import namedtuple
from copy import deepcopy
from slicerl.tools import m_lin_fit, pearson_distance

EventTuple = namedtuple("EventTuple", ["E", "x", "z", "cluster_idx", "pndr_idx",
                                       "mc_idx", "slicerl_idx"])

#======================================================================
class Event:
    """Event class keeping track of calohits in a 2D plane view."""
    
    #----------------------------------------------------------------------
    def __init__(self, calohits, min_hits=1, max_hits=15000):
        """
        Parameters
        ----------
            - calohits : np.array, shape=(7, num calohits), containing all the
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
        self.calohits = calohits

        # check if (E,x,z) inputs are in range
        assert np.all(calohits[0] < 500)
        assert np.all(np.logical_and(calohits[1] >= -0.37260447692861504,  calohits[1] <= 0.37260447692861504))
        assert np.all(np.logical_and(calohits[2] >= -0.35284, calohits[2] <= 0.91702))

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
        
        # point cloud to be passed to the agent:
        # (energy, x, z, pfo cluster idx, current status)
        point_cloud  = np.concatenate([calohits[:4], calohits[6:]])
        self.num_calohits = point_cloud.shape[1]

        # pad the point cloud according to max_calohits
        self.max_hits = max_hits
        # TODO: not block the training, but skip the event and raise a warning
        assert self.num_calohits < self.max_hits
        padding           = ((0,0),(0, self.max_hits - self.num_calohits))
        self.point_cloud  = np.pad(point_cloud, padding)

    #----------------------------------------------------------------------
    def __len__(self):
        return self.num_calohits

    #----------------------------------------------------------------------
    def state(self):
        """
        Return the observable state: padded point cloud, describing 
        (E, x, z, pfo cluster idx, current status) for each calohit
        
        Returns
        -------
            - array of shape=(5, max_hits)
        """
        return self.point_cloud

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
        array    = deepcopy(self.point_cloud[:, :self.num_calohits])
        array[0] = array[0] * 100  # to ADC
        array[1] = array[1] * 1000 # to cm
        array[2] = array[2] * 1000 # to cm

        # concatenate pndr idx row
        array = np.concatenate([
                        array[:4],               # (E, x, z, pfo cluster idx)
                        [self.pndr_idx],         # size pndr idx (original order)
                        [self.mc_idx],           # size mc idx   (original order)
                        array[-1:]               # status vector
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