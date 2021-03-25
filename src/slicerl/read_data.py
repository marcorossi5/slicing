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
    def __init__(self, infile, nmax = -1):
        """Initialize the reader."""
        self.infile = infile
        self.readline_fn = lambda x: np.array(next(x))[:-1].astype(np.float32)
        self.nmax = nmax
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
    def __init__(self, infile, nmax):
        self.reader = Reader(infile, nmax)

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
    def __init__(self, infile, nmax, min_hits):
        """
        Parameters
        ----------
            infile:   str, input file name
            nmax:     int, max number of events to load
            min_hits: int, consider slices with more than min_hits Calohits only
        """
        Image.__init__(self, infile, nmax)
        self.min_hits = min_hits
        self.printouts=10

    #----------------------------------------------------------------------
    def process(self, event):
        return Event(event, self.min_hits)

#======================================================================
def load_Events_from_file(filename, nev, R):
    """
    Utility function to load Jets object list from file. Return a list of Event
    objects. Loading into Events objects applies additional cuts in pT and
    rapidity on event particles.

    Parameters
    ----------
        - filename: str, file to load events from
        - nev:      int, number of events to load
        - R:        float, radius parameter of jet clustering algorithm
    
    Returns
    -------
        - list
            list of loaded Event object (with length equal to nev)
    """
    reader = Events(filename, nev, R)
    events = reader.values()
    return [Event(plane_view) for plane_view in events]

#======================================================================
def save_Event_list_to_file(events, filename):
    """
    Utility function to save Event.particles lists to file. Each line represents
    an Event, with all Particle objects contained in Event.particles. For each
    Particle, store 4-momentum and the additional information contained in PU
    and status attributes.

    Parameters
    ----------
        - event_list: list, list of Event objects
        - filename:   str
    """
    with gzip.open(filename, 'wb') as wfp:
        for event in events:
            event.dump(wfp)
