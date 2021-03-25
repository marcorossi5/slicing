# This file is part of SliceRL by M. Rossi

import sys
import numpy as np
import math
import json

#======================================================================
class Event:
    """Event class keeping track of calohits in a 2D plane view."""
    
    #----------------------------------------------------------------------
    def __init__(self, calohits, min_hits=0):
        """
        Parameters
        ----------
            - calohits : np.array, shape=(6, num calohits), containing all the
                         event information. Each calohit is described by:
                         energies, x coordinate, z coordinate, input pfo list
                         cluster idx, pandora slice index, cheating slice idx
                         (mc truth).
            - min_hits : int, consider only slices with more than min_hits         
        """
        self.calohits = []
        for i,c in enumerate(calohits.T):
            in_slice_idx = np.where(calohits[-1] == c[-1])[0]
            if len(in_slice_idx) >= min_hits:
                mcstate = (
                           calohits[0][in_slice_idx].sum(), # cumulative energy
                           calohits[0][in_slice_idx].sum(), # cumulative x
                           calohits[0][in_slice_idx].sum()  # cumulative z
                          )
                self.calohits.append(CaloHit(c[0], c[1], c[2],c[3], c[4], c[5],
                                     neighbours=in_slice_idx, mcstate=mcstate))

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
    def dump(self, fname):
        """ Dump particle list to fname file """
        data = [{'E': p.E, 'px': p.px, 'py': p.py, 'pz': p.pz, 'PU': p.PU, 'status': p.status} for p in self.particles]
        line = json.dumps(data, separators=(',', ':')).encode('utf-8')
        fname.write(line)
        fname.write(b'\n')

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
         - status:      the current slice index the calohit belongs to (changed
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
        self.status      = status

"""
TODO: rename Particle into CaloHit
"""