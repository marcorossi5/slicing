# This file is part of SliceRL by M. Rossi
import os
from tensorflow.keras.utils import Progbar
from slicerl.Event import Event
from slicerl.tools import quality_metric
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import math

#----------------------------------------------------------------------
"""
Color list. An exhaustive list of colors can be retrieved from matplotlib
printing matplotlib.colors.CSS4_COLORS.keys().
"""

colors = [
    'black', 'peru', 'deepskyblue', 'darkorchid', 'darkgoldenrod', 'teal',
    'dodgerblue', 'brown', 'darkslategrey', 'turquoise', 'lightsalmon', 'plum',
    'darkcyan', 'orange', 'slategrey', 'darkmagenta', 'limegreen', 'deeppink',
    'red', 'springgreen', 'midnightblue','green', 'mediumpurple',
    'mediumvioletred', 'dimgrey', 'blueviolet', 'lightskyblue', 'darksalmon',
    'royalblue', 'fuchsia', 'mediumaquamarine', 'mediumblue', 'grey', 'sienna',
    'mediumslateblue', 'seagreen', 'purple', 'greenyellow', 'darkviolet', 'coral',
    'darkblue', 'goldenrod', 'lime', 'cornflowerblue', 'darkturquoise', 'orangered',
    'cadetblue', 'lightcoral', 'skyblue', 'mediumseagreen', 'tomato', 'blue',
    'pink', 'olivedrab', 'rosybrown', 'darkseagreen', 'orchid', 'olive',
    'lightseagreen', 'cyan', 'dimgrey', 'magenta', 'darkolivegreen', 'slateblue',
    'lightgreen', 'navy', 'indianred', 'lawngreen', 'sandybrown', 'steelblue',
    'salmon', 'hotpink', 'darkgrey', 'violet',
    'cornflowerblue', 'snow', 'peru', 'dimgray', 'lightyellow',
    'indianred', 'palegoldenrod', 'darkgrey', 'mediumblue',
    'peachpuff', 'hotpink', 'green', 'brown', 'lightgoldenrodyellow',
    'mediumturquoise', 'lightslategrey', 'slateblue', 'purple',
    'ivory', 'lemonchiffon', 'orchid', 'darkred', 'chocolate',
    'aquamarine', 'cadetblue', 'thistle', 'orange', 'darkkhaki',
    'yellowgreen', 'lightsalmon', 'lightsteelblue', 'olivedrab',
    'mediumorchid', 'papayawhip', 'lime', 'gainsboro', 'teal', 'coral',
    'lightslategray', 'cyan', 'lightgrey', 'honeydew', 'mediumvioletred',
    'chartreuse', 'slategray', 'steelblue', 'gray', 'orangered',
    'mediumseagreen', 'aqua', 'rebeccapurple', 'saddlebrown',
    'lawngreen', 'powderblue', 'darkseagreen'
        ]
cmap = mpl.colors.ListedColormap(colors)
boundaries = np.arange(len(colors)+1) - 1.5
norm = mpl.colors.BoundaryNorm(boundaries, cmap.N, clip=True)

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
        nmc = np.array( [len(set(event.mc_idx)) for event in events] )
        nddpg = np.array( [len(set(event.slicerl_idx)) for event in events] )
    
    bins = np.linspace(0, 200, 201)
    hnmc, _   = np.histogram(nmc, bins=bins)
    hnddpg, _ = np.histogram(nddpg, bins=bins)

    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(18,14))
    
    plt.hist(bins[:-1], bins, weights=hnmc, histtype='step', color='blue', label='mc')
    plt.hist(bins[:-1], bins, weights=hnddpg, histtype='step', color='red', label='ddpg')

    plt.xlabel("multiplicity", loc='right')
    plt.xlim((bins[0], bins[-1]))
    plt.legend()
    fname = f"{output_folder}/multiplicity.pdf"
    plt.savefig(fname, bbox_inches='tight')
        
    print_stats('Slice multiplicity', nddpg  , nmc, output_folder=output_folder)
    resultsdir = '%s/results' % output_folder
    np.save(f"{resultsdir}/nmc.npy", nmc)
    np.save(f"{resultsdir}/nddpg.npy", nddpg)

#----------------------------------------------------------------------
def plot_EMD(events, events_obj, output_folder='./', loaddir=None):
    """Plot the energy movers distance between slices and output some statistics."""
    if loaddir is not None:
        fname = '%s/emdddpg.npy' % loaddir
        emdddpg = np.load(fname)
    else:
        emdddpg  = []
        for i, event in enumerate(events):
            num_clusters = event.slicerl_idx.max().astype(np.int32) + 1
            for idx in range(num_clusters):
                m = event.slicerl_idx == idx
                Es = event.E[m].sum()
                xs = event.x[m].sum()
                zs = event.z[m].sum()
                slice_state = np.array([Es, xs, zs])
                # get the index of the first calohit in the slice
                cidx = np.argwhere(m)[0,0]
                mc_state = events_obj[i].calohits[cidx].mc_idx
                emdddpg.append(quality_metric(slice_state, mc_state))
        emdddpg = np.array(emdddpg)
    
    bins = np.linspace(0, 200, 201)
    hemdddpg, _ = np.histogram(emdddpg, bins=bins)

    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(18,14))
    
    plt.hist(bins[:-1], bins, weights=hemdddpg, histtype='step', color='red')

    plt.xlabel("emd", loc='right')
    plt.xlim((bins[0], bins[-1]))
    fname = f"{output_folder}/emd.pdf"
    plt.savefig(fname, bbox_inches='tight')
        
    print_stats('Slice EMD', emdddpg, 0, output_folder=output_folder)
    resultsdir = '%s/results' % output_folder
    np.save(f"{resultsdir}/emdddpg.npy", emdddpg)

#----------------------------------------------------------------------
def plot_slice_size(events, output_folder='./', loaddir=None):
    """Plot the slice size distribution and output some statistics."""
    bins = np.linspace(0, 100, 101)

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

    plt.xlabel("size", loc='right')
    plt.xlim((bins[0], bins[-1]))

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
    num_rl_clusters = len(set(event_arr.slicerl_idx))
    # sort all the cluster with greater number of hits
    print(f"Plotting event with {len(event_arr.x)} calohits")
    print(f"Plotting {num_rl_clusters} slices in event")
    ax.scatter(event_arr.x, event_arr.z, s=0.5, c=event_arr.slicerl_idx, marker='.', cmap=cmap, norm=norm)
    ax.set_box_aspect(1)

    ax = fig.add_subplot(122)
    ax.set_title(f"Cheating Algorithm Truths, 2D plane view")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    num_mc_clusters = int(event_arr.mc_idx.max()) + 1
    print(f"Plotting {num_mc_clusters} true slices in event")
    ax.scatter(event_arr.x, event_arr.z, s=0.5, c=event_arr.mc_idx, marker='.', cmap=cmap, norm=norm)
    ax.set_box_aspect(1)
    fname = f"{output_folder}/pview.pdf"
    plt.savefig(fname, bbox_inches='tight')

#----------------------------------------------------------------------
def produce_slicing_animation(nmd, fname):
    """
    Produce an output animation to visualize event processing.
    
    Parameters
    ----------
        - nmd   : EventTuple, namedtuple from event
        - fname : str, output animation filename
    """
    plt.rcParams.update({'font.size': 10})
    fig = plt.figure(figsize=(6.4*2, 4.8))

    # at this point the mc_idx must already be sorted in increasing order
    # the agent must learn to capture the largest slice first
    # so i do not need to sort here
    # just plot all the colors stored in mc_idx
    ax = fig.add_subplot(122)
    ax.set_title(f"Cheating Algorithm Truths, 2D plane view")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    ax.scatter(nmd.x, nmd.z, s=0.5, c=nmd.mc_idx, marker='.', cmap=cmap, norm=norm)
    ax.set_box_aspect(1)

    num_clusters = len(set(nmd.slicerl_idx))

    # now plot the first frame with all black points
    # every new frame shows a new slice with a new color
    c = np.full_like(nmd.z, -1)

    sort_fn = lambda x: np.count_nonzero(nmd.slicerl_idx == x)
    s_idx = sorted(list(set(nmd.slicerl_idx)), key=sort_fn, reverse=True)

    ax = fig.add_subplot(121)
    ax.set_title(f"Slicing Algorithm Output, 2D plane view")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    scatt = ax.scatter(nmd.x, nmd.z, s=0.5, c=c, marker='.', cmap=cmap, norm=norm)
    ax.set_xlabel(f"Episode start")

    def animate(i, scatterplot):
        # FuncAnimation repeats twice the first frame, but skip that since it's
        # already built. First color in colors list is black.
        if i == -1:
            return (scatterplot,)
        idx = s_idx[i]
        m = nmd.slicerl_idx == s_idx[i]
        c[m] = i
        scatterplot.set_array(c)
        ax.xaxis.label.set_color(colors[i+1])
        ax.set_xlabel(f"Slice: {i}")
        return (scatterplot,) # must return an iterable

    anim = mpl.animation.FuncAnimation(fig, animate, frames=range(-1, num_clusters), interval=2000, fargs=(scatt,), blit=True)

    writergif = mpl.animation.PillowWriter(fps=1)
    anim.save(fname, writer=writergif)    

    # close figure
    plt.close()

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
def make_plots(events_obj, plotdir):
    """
    Make diagnostics plots from a list of subtracted events.

    Parameters
    ----------
        - events_obj:  list, list of sliced Event objects
        - plotdir:     str, plots output folder
    """
    events = [event.calohits_to_namedtuple() for event in events_obj]
    plot_multiplicity(events, plotdir)
    plot_slice_size(events, plotdir)
    plot_plane_view(events, plotdir)
    # plot_EMD(events, events_obj, plotdir)
    produce_slicing_animation(events[0], f"{plotdir}/slicing.gif")

#----------------------------------------------------------------------
def load_and_dump_plots(plotdir, loaddir):
    """
    Make diagnostics plots from plot data contained in loaddir.

    Parameters
    ----------
        - plotdir: str, plots output folder
        - loaddir: str, directory where to load plot data from
    """
    plot_multiplicity(None, plotdir, loaddir)
    plot_slice_size(None, plotdir, loaddir)
    plot_plane_view(None, plotdir, loaddir)
    # plot_EMD(None, None, plotdir, loaddir)

# TODO: fix plot_plane_view when loaded from results (it always needs the events)
# TODO: fix EMD function in this version