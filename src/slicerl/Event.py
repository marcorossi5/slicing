# This file is part of SliceRL by M. Rossi

import sys
import numpy as np
import math
import json, gzip
from collections import namedtuple
from copy import deepcopy

EventTuple = namedtuple("EventTuple", ["E", "x", "z", "cluster_idx", "pndr_idx",
                                       "mc_idx", "slicerl_idx"])

#======================================================================
class Event:
    """Event class keeping track of calohits in a 2D plane view."""
    
    #----------------------------------------------------------------------
    def __init__(self, calohits, k, min_hits=1):
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
            - Lneigh   : neighbourhood radius [10^1 m]
        """
        self.k = k
        self.calohits = []
        for i,c in enumerate(calohits.T):
            # find calohits indices belonging to the same slice
            in_slice_m = np.where(calohits[5] == c[5])[0]

            include = len(in_slice_m) >= min_hits
            mcstate = np.array(
                       [calohits[0][in_slice_m].sum(),  # cumulative energy
                        calohits[1][in_slice_m].sum(),  # cumulative x
                        calohits[2][in_slice_m].sum()]  # cumulative z
                      )
            # find the k closest calohits in (x,z) plane
            sqdist = (calohits[1] - c[1])**2 + (calohits[2] - c[2])**2
            idx = np.argpartition(sqdist,1+self.k)[:1+self.k]
            sorted_idx = idx[np.argsort(sqdist[idx])]
            neighbourhood = Graph(calohits[:, sorted_idx],
                                  sorted_idx,
                                  sqdist[sorted_idx])

            # calohits with status different from -1 or None will never be
            # processed. The model focuses on calohits with more calohits
            # than min_hits in the cluster, but the graph is allowed to
            # consider all the calohits.
            status = c[6] if include else -2.
            self.calohits.append(CaloHit(c[0], c[1], c[2],c[3], c[4], c[5],
                                    neighbourhood=neighbourhood,
                                    mc_neighbours_m=in_slice_m, mcstate=mcstate,
                                    status=status))

    #----------------------------------------------------------------------
    def __len__(self):
        return len(self.calohits)

    #----------------------------------------------------------------------
    def state(self, i):
        """
        Return the observable state of the calohit at index i.
        
        Returns
        -------
            - array of shape=(1+k,6)
        """
        if i < len(self.calohits):
            return self.calohits[i].neighbourhood.state(self.calohits)
        else:
            return np.zeros([1+self.k, 6])

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
            Calohit status attribute ranges in [0,128), but may contain holes.
            Also, in a slice is the subset of calohits that counts, not the
            value of the slice index. Then a reordering is applied before
            returning the array. To put slice indices in positional order.
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
        rows =  np.array(rows)

        # must order the status vector
        row = deepcopy(rows[6])
        unique_idx = set(row)
        count = 0
        for idx in unique_idx:
            m = row == idx
            rows[6][m] = count
            count += 1
        return rows
        


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

#======================================================================
class CaloHit:
    """Keep track of calohit properties."""

    #----------------------------------------------------------------------
    def __init__(self, E, x, z, clusterIdx=-1, pndrIdx=None, mcIdx=None, neighbourhood=None,
                 mc_neighbours_m=None, mcstate=None, status=None):
        """
        Create a calohit that has  E,x,z members as well as
         - E               : the energy carried by the calohit [10^-2 ADC]
         - x               : the drift coordinate of the calohit [10^1 m]
         - z               : the 2D plane coordinate of the calohit [10^1 m]
         - clusterIdx      : additional information from other pandora algorithms,
                             -1 if not provided
         - pndrIdx         : current pandora slicing algorithm output
         - mcIdx           : truth slice index
         - neighbourhood   : Graph object describing calohit neighbourhood
         - mc_neighbours_m : mask array of calohits from the same slice (mc truth) (TODO: remove this?)
         - mc_state        : the cumulative (E,x,z) info about the slice the calohit
                             belongs to (mc truth)
         - status          : the current slice index the calohit belongs to (changed
                             by the agent)
        """
        self.E               = E
        self.x               = x
        self.z               = z
        self.clusterIdx      = clusterIdx
        self.pndrIdx         = pndrIdx
        self.mcIdx           = mcIdx
        self.neighbourhood   = neighbourhood
        self.mc_neighbours_m = mc_neighbours_m if mc_neighbours_m is not None else []
        self.mc_state        = mcstate
        self.status          = status

#======================================================================
class Graph:
    """ Shape calohit neighbourhood into a graph. """

    #----------------------------------------------------------------------
    def __init__(self, calohits, idxs, sqdist):
        """
        Build the Graph object around a specific calohit.

        Parameters
        ----------
            - calohits : array of calohits feature in the event, shape=(7, 1+k)
            - idxs     : array, ordered neighbours indices, shape=(1+k,)
            - sqdist   : array of squared distances from central node, shape=(1+k,)
        
        Note
        ----
            all arrays are ordered by increasing calohit distance from center in
            the (x,z) plane
        """
        self.neighbours_idxs = idxs                     # neighbour calohits indices
        self.fv = np.concatenate([[sqdist],
                                  calohits[(0,1,2),:]]) # feature vector
        self.cv = calohits[3,:]                         # cluster idx vector

    #----------------------------------------------------------------------
    def state(self, calohits):
        """
        Return the graph node feature vector concatenated with the current value
        of Calohit status attribute of calohits in the neighbourhood.
        
        Parameter
        ---------
            - calohits : list of Calohit objects in the neighbourhood
        
        Returns
        -------
            array of shape=(1+k, 6)
        """
        sv = np.array( [calohits[idx].status for idx in self.neighbours_idxs] )
        return np.concatenate([self.fv, [self.cv/100], [sv/100]]).T

# TODO: think about normalizing the cluster_idx and status inputs to the actor model