# This file is part of SliceRL by M. Rossi

import sys
import numpy as np
import math
import json, gzip
from collections import namedtuple

EventTuple = namedtuple("EventTuple", ["E", "x", "z", "cluster_idx", "pndr_idx",
                                       "mc_idx", "slicerl_idx"])

#======================================================================
class Event:
    """Event class keeping track of calohits in a 2D plane view."""
    
    #----------------------------------------------------------------------
    def __init__(self, calohits, min_hits=1):
        """
        Parameters
        ----------
            - calohits : np.array, shape=(7, num calohits), containing all the
                         event information. Each calohit is described by:
                            - energy
                            - x coordinate
                            - z coordinate
                            - input pfo list cluster idx
                            - pandora slice index
                            - cheating slice idx (mc truth)
                            - slicing output (array of minus ones if not loading
                              inference results)
            - min_hits : int, consider only slices with equal or more than min_hits         
        """
        self.calohits = []
        for i,c in enumerate(calohits.T):
            # find calohits indices belonging to the same slice
            in_slice_idx = np.where(calohits[5] == c[5])[0]
            if len(in_slice_idx) >= min_hits:
                mcstate = np.array(
                           [calohits[0][in_slice_idx].sum(),  # cumulative energy
                            calohits[0][in_slice_idx].sum(),  # cumulative x
                            calohits[0][in_slice_idx].sum()]  # cumulative z
                          )
                self.calohits.append(CaloHit(c[0], c[1], c[2],c[3], c[4], c[5],
                                     neighbours=in_slice_idx, mcstate=mcstate,
                                     status=c[6]))
        self.slices_info = []


    #----------------------------------------------------------------------
    def __len__(self):
        return len(self.calohits)

    #----------------------------------------------------------------------
    def state(self, i):
        """Return the observable state of the calohit at index i."""
        if i < len(self.calohits):
            return np.array([
                        self.calohits[i].E, self.calohits[i].x,
                        self.calohits[i].z, self.calohits[i].clusterIdx
                           ])
        else:
            return np.zeros(4)
            # state = [E, x, z, cluster]
        #     return ( np.array([
        #                 self.calohits[i].E, self.calohits[i].x, self.calohits[i].z
        #                     ]),\
        #              np.array([self.calohits[i].clusterIdx])
        #            )
        # else:
        #     return ( np.zeros(3), -1*np.ones(1) )


    #----------------------------------------------------------------------
    def calohits_to_array(self):
        """
        Takes calohits list into a list of arrays for plot rendering.

        Returns
        ----------
            - numpy array of shape (7, num calohits). Rows contain in order:
              energies, xs, zs, cluster_idx, pndr_idx, cheating_idx, slicerl_idx

        """
        rows = [[] for i in range(7)]
        for c in self.calohits:
            rows[0].append(c.E*100)
            rows[1].append(c.x*1000)
            rows[2].append(c.z*1000)
            rows[3].append(c.clusterIdx)
            rows[4].append(c.pndrIdx)
            rows[5].append(c.mcIdx)
            rows[6].append(c.status)
        return np.array(rows)

    #----------------------------------------------------------------------
    def calohits_to_namedtuple(self):
        """
        Takes calohits list into an EventTuple object for plot rendering.

        Returns
        ----------
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

#======================================================================
class CaloHit:
    """Keep track of calohit properties."""

    #----------------------------------------------------------------------
    def __init__(self, E, x, z, clusterIdx=-1, pndrIdx=None, mcIdx=None,
                 neighbours=None, mcstate=None, status=None):
        """
        Create a calohit that has  E,x,z members as well as
         - E:          the energy carried by the calohit [10^-2 ADC]
         - x:          the drift coordinate of the calohit [10^1 m]
         - z:          the 2D plane coordinate of the calohit [10^1 m]
         - clusterIdx: additional information from other pandora algorithms,
                       -1 if not provided
         - mcIdx:      truth slice index
         - pndrIdx:    current pandora slicing algorithm output
         - neighbours: indices array of calohits from the same slice (mc truth)
         - mc_state:   the cumulative (E,x,z) info about the slice the calohit
                       belongs to (mc truth)
         - status:     the current slice index the calohit belongs to (changed
                       by the agent)
        """
        self.E          = E
        self.x          = x
        self.z          = z
        self.clusterIdx = clusterIdx
        self.pndrIdx    = pndrIdx
        self.mcIdx      = mcIdx
        self.neighbours = neighbours if neighbours is not None else []
        self.mc_state   = mcstate
        self.status     = status

"""
TODO: rename Particle into CaloHit
"""