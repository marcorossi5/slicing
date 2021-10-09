# This file is part of SliceRL by M. Rossi
from slicerl.Event import Event

import json, gzip, sys, csv
from abc import ABC, abstractmethod
import numpy as np

# ======================================================================
class Reader(object):
    """
    Reader for files consisting of a sequence of json objects.
    Any pure string object is considered to be part of a header (even if it appears at the end!)
    """

    # ----------------------------------------------------------------------
    def __init__(self, infile, nmax=-1, load_results=False):
        """Initialize the reader."""
        self.infile = infile
        self.readline_fn = lambda x: np.array(next(x))[:-1].astype(np.float32)
        self.nmax = nmax
        self.load_results = load_results
        self.reset()

    # ----------------------------------------------------------------------
    def reset(self):
        """
        Reset the reader to the start of the file, clear the header and event count.
        """
        self.stream = csv.reader(gzip.open(self.infile, "rt"))
        self.n = 0
        self.header = []

    # ----------------------------------------------------------------------
    def __iter__(self):
        # needed for iteration to work
        return self

    # ----------------------------------------------------------------------
    def __next__(self):
        ev = self.next_event()
        if ev is None:
            raise StopIteration
        else:
            return ev

    # ----------------------------------------------------------------------
    def next(self):
        return self.__next__()

    # ----------------------------------------------------------------------
    def next_plane(self):
        """ Main function to read one plane view at a time from file."""
        try:
            c = []
            c.append(self.readline_fn(self.stream) / 100)  # energies [ADC]/100
            c.append(self.readline_fn(self.stream) / 1000.0)  # xs [10^1 m]
            c.append(self.readline_fn(self.stream) / 1000.0)  # zs [10^1 m]
            c.append(self.readline_fn(self.stream) / 1000.0)  # x expected direction
            c.append(self.readline_fn(self.stream) / 1000.0)  # z expected direction
            c.append(self.readline_fn(self.stream))  # cluster_idx
            c.append(self.readline_fn(self.stream))  # pndr_idx
            c.append(self.readline_fn(self.stream))  # cheating_idx (mc truth)
            if self.load_results:
                c.append(self.readline_fn(self.stream))  # slicerl_idx
            else:
                c.append(np.full_like(c[-1], -1))
            c.append(self.readline_fn(self.stream))  # test beam flag
            c.append(self.readline_fn(self.stream))  # PDG
            c.append(self.readline_fn(self.stream))  # pfo index
        except IOError:
            print(
                "# got to end with IOError (maybe gzip structure broken?) around event",
                self.n,
                file=sys.stderr,
            )
            return None
        except EOFError:
            print(
                "# got to end with EOFError (maybe gzip structure broken?) around event",
                self.n,
                file=sys.stderr,
            )
            return None
        except ValueError:
            print(
                "# got to end with ValueError (empty json entry) around event",
                self.n,
                file=sys.stderr,
            )
            return None
        except StopIteration:
            print(f"# Exiting after having read all the {self.n} events")
            return None
        return np.stack(c)

    # ----------------------------------------------------------------------
    def next_event(self):
        """ Main function to read one plane view at a time from file."""
        # return after hitting the maximum number of events
        if self.n == self.nmax:
            print(f"# Exiting after having read {self.nmax} events")
            return None
        tpc_view_U = self.next_plane()
        if tpc_view_U is not None:
            tpc_view_V = self.next_plane()
            if tpc_view_V is None:
                raise EOFError(f"Missing TPC plane view V at event {self.n}")
            tpc_view_W = self.next_plane()
            if tpc_view_W is None:
                raise EOFError(f"Missing TPC plane view W at event {self.n}")
            self.n += 1
            return tpc_view_U, tpc_view_V, tpc_view_W
        return None


# ======================================================================
class Image(ABC):
    """Image which transforms point-like information into pixelated 2D
    images which can be processed by convolutional neural networks."""

    def __init__(self, infile, nmax, load_results=False):
        self.reader = Reader(infile, nmax, load_results)

    # ----------------------------------------------------------------------
    @abstractmethod
    def process(self, event):
        pass

    # ----------------------------------------------------------------------
    def values(self):
        res = []
        while True:
            tpc_views = self.reader.next_event()
            if tpc_views is not None:
                res.append(self.process(tpc_views))
            else:
                break
        self.reader.reset()
        return res


# ======================================================================
class Events(Image):
    """Read input file with calohits and transform into python events."""

    # ----------------------------------------------------------------------
    def __init__(self, infile, nmax, min_hits, max_hits, load_results=False):
        """
        Parameters
        ----------
            infile       : str, input file name
            nmax         : int, max number of events to load
            min_hits     : int, consider slices with more than min_hits Calohits only
            load_results : bool, wether to load results from a previous slicing
                           inference or not.
        """
        Image.__init__(self, infile, nmax, load_results)
        self.min_hits = min_hits
        self.max_hits = max_hits
        self.printouts = 10

    # ----------------------------------------------------------------------
    def process(self, tpc_views):
        # order by increasing x
        return Event(tpc_views, self.min_hits)


# ======================================================================
def load_Events_from_file(
    filename, nev=-1, min_hits=1, max_hits=15000, load_results=False
):
    """
    Utility function to load Events object from file. Return a list of Event
    objects.

    Parameters
    ----------
        - filename     : str, file to load events from
        - nev          : int, number of events to load
        - min_hits     : int, consider slices with more than min_hits Calohits only
        - max_hits     : int, max hits to be processed by network
        - load_results : bool, wether to load results from a previous slicing
                           inference or not.

    Returns
    -------
        - list
            list of loaded Event object (with length equal to nev)
    """
    print(f"[+] Reading {filename}")
    reader = Events(filename, nev, min_hits, max_hits, load_results)
    return reader.values()


# ======================================================================
def load_Events_from_files(
    filelist, nev=-1, min_hits=1, max_hits=15000, load_results=False
):
    """
    Utility function to load Events object from file. Return a list of Event
    objects.

    Parameters
    ----------
        - filename     : list, list of files to load events from
        - nev          : int, number of events to load
        - min_hits     : int, consider slices with more than min_hits Calohits only
        - max_hits     : int, max hits to be processed by network
        - load_results : bool, wether to load results from a previous slicing
                           inference or not.

    Returns
    -------
        - list
            list of loaded Event object (with length equal to nev)
    """
    events = []
    for fname in filelist:
        print(f"[+] Reading {fname}")
        events.extend(Events(fname, nev, min_hits, max_hits, load_results).values())
    return events


# ======================================================================
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
    with gzip.open(filename, "wb") as wfp:
        for event in events:
            event.dump(wfp)
