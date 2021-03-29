# This file is part of SliceRL by M. Rossi
import os
from tensorflow.keras.utils import Progbar
from expurl.read_data import Jets
from expurl.Event import Event
from expurl.tools import jet_emd, confusion_matrix_per_event
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from copy import deepcopy
import fastjet as fj
import math

#----------------------------------------------------------------------
colors = [
    "lightsalmon",
    "orange",
    "springgreen",
    "fuchsia",
    "lime",
    "lightcoral",
    "pink",
    "darkseagreen",
    "gold",
    "red",
    "deepskyblue",
    "lightgreen",
    "coral",
    "aqua",
    "lightgreen",
    "mediumaquamarine"
]
l = len(colors)

#----------------------------------------------------------------------
def print_stats(name, data, mass_ref, output_folder='./'):
    """Print statistics on the mass distribution."""
    r_plain = np.array(data)-mass_ref
    m = np.median(r_plain)
    a = np.mean(r_plain)
    s = np.std(r_plain)
    fname = f"{output_folder}/diagnostics.txt"
    with open(fname,'a+') as f:
        print(f"{name:<25}:\tmedian-diff {m:.2f}\tavg-diff {a:.2f}\tstd-diff {s:.2f}",
              file=f)

#----------------------------------------------------------------------
def plot_multiplicity(events, output_folder='./', loaddir=None):
    """Plot the slice multiplicity distribution and output some statistics."""
    if loaddir is not None:
        fname = '%s/nmc.npy' % loaddir
        nmc = np.load(fname)
        fname = '%s/nddpg.npy' % loaddir
        nddpg = np.load(fname)
    else:
        nmc = np.array( [int(event.mc_idx.max()) + 1 for event in events] )
        nddpg = np.array( [int(event.slicerl_idx.max()) + 1 for event in events] )
    
    bins = np.linspace(-10000, 10000, 201)
    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(18,14))
    counts, _, _ = plt.hist(nddpg-nmc, bins=bins, alpha=0.5,
             linestyle='dotted', facecolor='lawngreen', label='DDPG-Slicing')
    plt.hist(bins[:-1], bins, weights=counts, histtype='step', color='green', linestyle='dotted')

    plt.xlabel("$n_{reco}- n_{mc}$", loc='right')
    plt.xlim((bins[0], bins[-1]))
    plt.legend()
    fname = f"{output_folder}/multiplicity.pdf"
    plt.savefig(fname, bbox_inches='tight')
        
    print_stats('Slice multiplicity', nddpg  , nmc, output_folder=output_folder)
    resultsdir = '%s/results' % output_folder
    np.save(f"{resultsdir}/nmc.npy", nmc)
    np.save(f"{resultsdir}/nddpg.npy", nddpg)

#----------------------------------------------------------------------
def plot_slice_size(events, output_folder='./', loaddir=None):
    """Plot the slice size distribution and output some statistics."""
    bins = np.linspace(0, 1000, 1001)

    if loaddir is not None:
        fname = '%s/smc.npy' % loaddir
        smc = np.load(fname)
        fname = '%s/sddpg.npy' % loaddir
        sddpg = np.load(fname)
    else:
        binc_mc = [np.bincount(event.mc_idx.astype(np.int32)[event.mc_idx >= 0]) for event in events]
        binc_ddpg = [np.bincount(event.slicerl_idx.astype(np.int32)) for event in events]
        smc = sum( [np.histogram(bc, bins=bins)[0] for bc in binc_mc] )
        sddpg = sum( [np.histogram(bc, bins=bins)[0] for bc in binc_ddpg] )
    
    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(18,14))

    plt.hist(bins[:-1], bins, weights=smc, histtype='step', color='blue', label='mc')
    plt.hist(bins[:-1], bins, weights=sddpg, histtype='step', color='red', label='ddpg')

    plt.xlabel("$s_{reco}- s_{mc}$", loc='right')
    plt.xlim((bins[0], bins[-1]))

    plt.yscale("log")
    plt.legend()
    fname = f"{output_folder}/slice_size.pdf"
    plt.savefig(fname, bbox_inches='tight')
        
    print_stats('Slice multiplicity', sddpg  , smc, output_folder=output_folder)
    resultsdir = '%s/results' % output_folder
    np.save(f"{resultsdir}/smc.npy", smc)
    np.save(f"{resultsdir}/sddpg.npy", sddpg)

#----------------------------------------------------------------------
def plot_plane_view(events, output_folder='./', loaddir=None):
    event_arr = events[0]

    fig = plt.figure(figsize=(18*2,14))
    ax = fig.add_subplot(121)
    ax.set_title(f"Slicing Algorithm Output, 2D plane view")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    num_clusters = int(event_arr.slicerl_idx.max()) + 1
    sort_fn = lambda x: np.count_nonzero( event_arr.slicerl_idx == x )
    sorted_indices = sorted( range(num_clusters), key=sort_fn, reverse=True)
    # sort all the cluster with greater number of hits
    print(f"Plotting {num_clusters} Slices in event over {event_arr.slicerl_idx.shape[0]} particles")
    for index in sorted_indices[:min(len(sorted_indices), 200)]:
        m = event_arr.slicerl_idx == index
        ax.scatter(event_arr.x[m], event_arr.z[m], marker='.', color=colors[index%l])
    ax.set_box_aspect(1)

    ax = fig.add_subplot(122)
    ax.set_title(f"Cheating Algorithm Truths, 2D plane view")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    num_clusters = int(event_arr.mc_idx.max()) + 1
    print(f"Plotting {num_clusters} true Slices in event")
    for index in range(num_clusters):
        m = event_arr.mc_idx == index
        ax.scatter(event_arr.x[m], event_arr.z[m], marker='.', color=colors[index%l])
    ax.set_box_aspect(1)
    fname = f"{output_folder}/pview.pdf"
    plt.savefig(fname, bbox_inches='tight')

#----------------------------------------------------------------------
def plot_ROC(scores, output_folder='./', loaddir=None):
    """Plot the ROC curve for particle classification and output some statistics."""
    # particle: list of np.arrays of shape=(num particles, 2) for all events
    if loaddir is not None:
        fname = '%s/confusion.npy' % loaddir
        tp, fp, fn, tn = np.load(fname)
    else:
        tp, fp, fn, tn = confusion_matrix_per_event(scores)
    efficiency = tp / (tp + fn)
    irejection = fp / (tn + fp)

    sqrtn = math.sqrt(len(tp))
    tp_avg = tp.mean() * 100
    tp_unc = tp.std() / sqrtn * 100
    fn_avg = fn.mean() * 100
    fn_unc = fn.std() / sqrtn * 100
    fp_avg = fp.mean() * 100
    fp_unc = fp.std() / sqrtn * 100
    tn_avg = tn.mean() * 100
    tn_unc = tn.std() / sqrtn * 100
    PU_unc = np.sqrt(tp_unc**2 + fn_unc**2)
    LV_unc = np.sqrt(tn_unc**2 + fp_unc**2)

    print(f"Dataset balancing: \t PU particles: {tp_avg+fn_avg:.2f}+-{PU_unc:.2f} % \t LV particles: {tn_avg+fp_avg:.2f}+-{LV_unc:.2f} % ")
    print("Average confusion matrix:")
    print( "\t\  true |        |        |")
    print( "\t  ----  |   PU   |   LV   |")
    print( "\t pred  \|        |        |")
    print( "\t---------------------------")
    print(f"\t   PU   |{tp_avg:>5.2f} % |{fp_avg:>5.2f} % |")
    print( "\t---------------------------")
    print(f"\t   LV   |{fn_avg:>5.2f} % |{tn_avg:>5.2f} % |")
    print( "\t---------------------------")

    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(18,14))
    plt.scatter(efficiency, irejection, marker='.', color="blue", label="DQN-Subtracting")

    plt.xlabel("Efficiency $\epsilon$", loc='right')
    plt.ylabel("$1/R$", loc="top")
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.legend()
    fname = f"{output_folder}/ROC.pdf"
    plt.savefig(fname, bbox_inches='tight')
        
    print_stats('Efficiency' , efficiency   , 0, output_folder=output_folder)
    print_stats('1/Rejection', irejection   , 0, output_folder=output_folder)
    resultsdir = '%s/results' % output_folder
    np.save(f"{resultsdir}/confusion.npy", np.stack([tp, fp, fn, tn]))

#----------------------------------------------------------------------
def inference(slicer, events):
    """
    Slice calohits from a list of Events objects. Returns the list of
    processed Events.

    Parameters
    ----------
        slicer: Slicer object
    
    Returns
    -------
        The list of subtracted Events.
    """
    progbar = Progbar(len(events))
    for i, event in enumerate(events):
        slicer(event)
        progbar.update(i+1)
    return events

#----------------------------------------------------------------------
def make_plots(events, plotdir):
    """
    Make diagnostics plots from a list of subtracted events.

    Parameters
    ----------
        - events:  list, list of sliced Event objects
        - plotdir: str, plots output folder
    """
    events = [event.calohits_to_namedtuple() for event in events]
    plot_multiplicity(events, plotdir)
    plot_slice_size(events, plotdir)
    plot_plane_view(events, plotdir)

#----------------------------------------------------------------------
def load_and_dump_plots(plotdir, loaddir):
    """
    Make diagnostics plots from plot data contained in loaddir.

    Parameters
    ----------
        - plotdir: str, plots output folder
        - loaddir: str, directory where to load plot data from
    """
    events = [event.calohits_to_namedtuple() for event in events]
    plot_multiplicity(None, plotdir, loaddir)
    plot_slice_size(None, plotdir, loaddir)
    plot_plane_view(None, plotdir, loaddir)
