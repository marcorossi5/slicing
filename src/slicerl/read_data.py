# This file is part of SliceRL by M. Rossi

import json, gzip, sys, csv
from abc import ABC, abstractmethod
import numpy as np
from math import log, ceil, floor, pi
from slicerl.Event import Event, CaloHit

#======================================================================
class Reader(object):
    """
    Reader for files consisting of a sequence of json objects.
    Any pure string object is considered to be part of a header (even if it appears at the end!)
    """

    #----------------------------------------------------------------------
    def __init__(self, infile, nmax=-1, num_lines=6):
        """Initialize the reader."""
        self.infile = infile
        self.readline_fn = lambda x: np.array(next(x))[:-1].astype(np.float32)
        self.nmax = nmax
        self.num_lines = num_lines
        self.reset()

    #----------------------------------------------------------------------
    def reset(self):
        """
        Reset the reader to the start of the file, clear the header and event count.
        """
        self.stream = csv.reader( gzip.open(self.infile,'rt') )
        self.n = 0
        self.header = []        
        
    #----------------------------------------------------------------------
    def __iter__(self):
        # needed for iteration to work 
        return self

    #----------------------------------------------------------------------
    def __next__(self):
        ev = self.next_event()
        if (ev is None): raise StopIteration
        else           : return ev

    #----------------------------------------------------------------------
    def next(self): return self.__next__()

    #----------------------------------------------------------------------
    def next_event(self):
        # we have hit the maximum number of events
        if (self.n == self.nmax):
            print ("# Exiting after having read nmax events")
            return None
        
        try:
            c = []
            c.append( self.readline_fn(self.stream) / 100 ) # energies [ADC]/100
            c.append( self.readline_fn(self.stream) / 1000.) # xs [10^1 m]
            c.append( self.readline_fn(self.stream) / 1000.) # zs [10^1 m]
            c.append( self.readline_fn(self.stream) ) # cluster_idx
            c.append( self.readline_fn(self.stream) ) # pndr_idx
            c.append( self.readline_fn(self.stream) ) # cheating_idx (mc truth)
            if self.num_lines == 7:
                c.append( self.readline_fn(self.stream) ) # slicerl_idx
            else:
                c.append( -1*np.ones_like(c[-1]) )
        except IOError:
            print("# got to end with IOError (maybe gzip structure broken?) around event", self.n, file=sys.stderr)
            return None
        except EOFError:
            print("# got to end with EOFError (maybe gzip structure broken?) around event", self.n, file=sys.stderr)
            return None
        except ValueError:
            print("# got to end with ValueError (empty json entry) around event", self.n, file=sys.stderr)
            return None
        except StopIteration:
            print("# Exiting after having read all the events")
            return None

        self.n += 1
        return np.stack(c)

#======================================================================
class Image(ABC):
    """Image which transforms point-like information into pixelated 2D
    images which can be processed by convolutional neural networks."""
    def __init__(self, infile, nmax, num_lines=6):
        self.reader = Reader(infile, nmax, num_lines)

    #----------------------------------------------------------------------
    @abstractmethod
    def process(self, event):
        pass

    #----------------------------------------------------------------------
    def values(self):
        res = []
        while True:
            event = self.reader.next_event()
            if event is not None:
                res.append(self.process(event))
            else:
                break
        self.reader.reset()
        return res

#======================================================================
class Events(Image):
    """Read input file with calohits and transform into python events."""

    #----------------------------------------------------------------------
    def __init__(self, infile, nmax, k, min_hits, num_lines=None):
        """
        Parameters
        ----------
            infile    : str, input file name
            nmax      : int, max number of events to load
            k         : the number of neighbours calohits to include in the Graph
            min_hits  : int, consider slices with more than min_hits Calohits only
            num_lines : int, number of lines stored in the file. Specifies if the
                        file contains inference results or not.
        """
        Image.__init__(self, infile, nmax, num_lines)
        self.min_hits = min_hits
        self.printouts=10
        self.k = k

    #----------------------------------------------------------------------
    def process(self, event):
        return Event(event, self.k, self.min_hits)

#======================================================================
def load_Events_from_file(filename, nev, k, min_hits=1, num_lines=6):
    """
    Utility function to load Events object from file. Return a list of Event
    objects.

    Parameters
    ----------
        - filename  : str, file to load events from
        - nev       : int, number of events to load
        - k         : the number of neighbours calohits to include in the Graph
        - min_hits  : int, consider slices with more than min_hits Calohits only
        - num_lines : int, number of lines stored in the file. Lines stand for:
                      energies, xs, zs, cluster_idx, pndr_idx, cheating_idx,
                      slicerl_idx (optional). 
    
    Returns
    -------
        - list
            list of loaded Event object (with length equal to nev)
    """
    reader = Events(filename, nev, k, min_hits, num_lines)
    return reader.values()

#======================================================================
def save_Event_list_to_file(events, filename):
    """
    Utility function to save Event.particles lists to file. Each line represents
    an Event, with all Particle objects contained in Event.particles. For each
    Particle, store 4-momentum and the additional information contained in PU
    and status attributes.

    Parameters
    ----------
        - events   : list, list of Event objects
        - filename : str
    """
    with gzip.open(filename, 'wb') as wfp:
        for event in events:
            event.dump(wfp)
