# This file is part of SliceRL by M. Rossi
import os
from tensorflow.keras.utils import Progbar
from slicerl.Event import Event
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import math
from time import time as tm

#----------------------------------------------------------------------
# available cmaps
"""
Color list. An exhaustive list of colors can be retrieved from matplotlib
printing matplotlib.colors.CSS4_COLORS.keys().
"""

colors = [
    'black', 'deepskyblue', 'peru', 'darkorchid', 'darkgoldenrod', 'teal',
    'dodgerblue', 'brown', 'darkslategrey', 'turquoise', 'lightsalmon', 'plum',
    'darkcyan', 'orange', 'slategrey', 'darkmagenta', 'limegreen', 'deeppink',
    'gold', 'springgreen', 'midnightblue','green', 'mediumpurple',
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
    'lawngreen', 'powderblue', 'darkseagreen', 'red'
        ]
cmap = mpl.colors.ListedColormap(colors)
boundaries = np.arange(len(colors)+1) - 1.5
norm = mpl.colors.BoundaryNorm(boundaries, cmap.N, clip=True)

l = len(colors)

vcmap = 'plasma'
vnorm = mpl.colors.Normalize(vmin=0., vmax=1.)

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
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("z [cm]")
    num_rl_clusters = len(set(event_arr.slicerl_idx))
    m = event_arr.slicerl_idx != -1
    # print(f"Hits not labelled: {np.count_nonzero(~m)}")
    # sort all the cluster with greater number of hits
    print(f"Plotting event with {len(event_arr.x)} calohits")
    print(f"Plotting {num_rl_clusters} slices in event")
    ax.scatter(event_arr.x[m], event_arr.z[m], s=1, c=event_arr.slicerl_idx[m], marker='.', cmap=cmap, norm=norm)
    ax.set_box_aspect(1)

    ax = fig.add_subplot(122)
    ax.set_title(f"Cheating Algorithm Truths, 2D plane view")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("z [cm]")
    num_mc_clusters = int(event_arr.mc_idx.max()) + 1
    print(f"Plotting {num_mc_clusters} true slices in event")
    ax.scatter(event_arr.x, event_arr.z, s=1, c=event_arr.mc_idx, marker='.', cmap=cmap, norm=norm)
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
    
    # now plot the first frame with all black points
    # every new frame shows a new slice with a new color
    sort_fn    = lambda x: np.count_nonzero(nmd.slicerl_idx == x)
    s_idx      = sorted(list(set(nmd.slicerl_idx)), key=sort_fn, reverse=True)
    nslices    = len(s_idx)

    sort_fn    = lambda x: np.count_nonzero(nmd.mc_idx == x)
    mc_idx     = sorted(list(set(nmd.mc_idx)), key=sort_fn, reverse=True)
    # nmcslices  = len(mc_idx)

    ax0 = fig.add_subplot(121)
    ax0.set_title(f"Slicing Algorithm Output, 2D plane view")
    ax0.set_xlabel("x [cm]")
    ax0.set_ylabel("z [cm]")
    scatt0 = ax0.scatter(nmd.x, nmd.z, s=1, c=nmd.slicerl_idx, marker='.', cmap=cmap, norm=norm)
    ax0.set_xlabel(f"Episode start")

    ax1 = fig.add_subplot(122)
    ax1.set_title(f"Cheating Algorithm Truths, 2D plane view")
    ax1.set_xlabel("x [cm]")
    ax1.set_ylabel("z [cm]")
    scatt1 = ax1.scatter(nmd.x, nmd.z, s=1, c=nmd.mc_idx, marker='.', cmap=cmap, norm=norm)
    ax1.set_box_aspect(1)

    def animate(i, scatterplot0, scatterplot1):
        # FuncAnimation repeats twice the first frame, but skip that since it's
        # already built. First color in colors list is black.
        if i == -1:
            return (scatterplot0, scatterplot1)
        # if i == 0:
        #     scatterplot0.set_cmap(vcmap)
        #     scatterplot0.set_norm(vnorm)
        #     scatterplot1.set_cmap(vcmap)
        #     scatterplot1.set_norm(vnorm)
        idx = s_idx[i]
        m = nmd.slicerl_idx == idx
        # c[m] = idx
        # scatterplot0.set_array(c)
        scatterplot0.set_array(np.where(m, 1, 0))
        ax0.xaxis.label.set_color(colors[int(idx)+1])
        ax0.set_xlabel(f"z [cm]    Slice: {int(idx)}")

        idx = mc_idx[i]
        m = nmd.mc_idx == idx
        # c[m] = idx
        # scatterplot0.set_array(c)
        scatterplot1.set_array(np.where(m, 1, 0))
        ax1.xaxis.label.set_color(colors[int(idx)+1])
        ax1.set_xlabel(f"z [cm]    Slice: {int(idx)}")

        return (scatterplot0, scatterplot1) # must return an iterable

    anim = mpl.animation.FuncAnimation(fig, animate, frames=range(-1, nslices), interval=5000, fargs=(scatt0, scatt1), blit=True)

    writergif = mpl.animation.PillowWriter(fps=0.5)
    anim.save(fname, writer=writergif)    

    # close figure
    plt.close()

#----------------------------------------------------------------------
def render(event, action_scores, fname):
    """
    Produce an output animation to visualize event processing. On the left
    subsequent actions. On the right the current mc slice. Start frame is just
    scatterplot without colors.
    
    Parameters
    ----------
        - event         : Event
        - action_scores : np.array, shape=(num_slices, num_calohits)
        - fname         : str, output animation filename
    """
    x = event.calohits[1] * 1000 # restore original unit measures [cm]
    z = event.calohits[2] * 1000 # restore original unit measures [cm]
    mc_idx = event.ordered_mc_idx
    null_score = np.zeros_like(action_scores[0])

    plt.rcParams.update({'font.size': 10})
    fig = plt.figure(figsize=(6.4*2, 4.8))

    # num_slices = max(len(set(mc_idx)), action_scores.shape[0])
    num_slices = len(set(mc_idx))
    if num_slices > action_scores.shape[0]:
        pad = np.zeros([num_slices - action_scores.shape[0], action_scores.shape[1]])
        action_scores = np.concatenate([action_scores, pad])

    # now plot the first frame with all black points
    # every new frame shows a new slice with a new color
    c = np.zeros_like(z)

    ax0 = fig.add_subplot(121)
    ax0.set_title("Slicing Algorithm Output, 2D plane view")
    ax0.set_xlabel("x [cm]")
    ax0.set_ylabel("z [cm]")
    scatt0 = ax0.scatter(x, z, s=1, c=c, marker='.', norm=vnorm)
    ax0.set_xlabel("Episode start")
    ax0.set_box_aspect(1)

    # at this point the mc_idx must already be sorted in increasing order
    # just plot all the colors stored in mc_idx
    ax1 = fig.add_subplot(122)
    ax1.set_title("Cheating Algorithm Truths, 2D plane view")
    ax1.set_xlabel("x [cm]")
    ax1.set_ylabel("z [cm]")
    scatt1 = ax1.scatter(x, z, s=1, c=mc_idx, marker='.', cmap=cmap, norm=norm)
    ax1.set_xlabel("MC slices")
    ax1.set_box_aspect(1)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.15, 0.01, 0.7])
    fig.colorbar(mpl.cm.ScalarMappable(norm=vnorm, cmap=vcmap), cax=cbar_ax)
    # fig.colorbar(z, cax=cbar_ax)

    def animate(i, scatterplot0, scatterplot1):
        # FuncAnimation repeats twice the first frame, but skip that since it's
        # already built. First color in colors list is black.
        if i == -1:
            return (scatterplot0, scatterplot1)
        scatterplot0.set_array(action_scores[i])
        ax0.set_xlabel(f"Slice: {i}")

        # plot the mask of the current mc slice
        
        scatterplot1.set_array((mc_idx == i).astype(np.int16))
        ax1.set_xlabel(f"Slice: {i}")
        if i == 0:
            scatterplot0.set_cmap(vcmap)
            scatterplot1.set_cmap(vcmap)
            scatterplot1.set_norm(vnorm)
        return (scatterplot0, scatterplot1) # must return an iterable

    anim = mpl.animation.FuncAnimation(fig, animate, frames=range(-1, num_slices), interval=2000, fargs=(scatt0, scatt1), blit=True)

    writergif = mpl.animation.PillowWriter(fps=1)
    anim.save(fname, writer=writergif)    

    # close figure
    plt.close()

#----------------------------------------------------------------------
def inference(slicer, events, visualize=False, gifname=None):
    """
    Slice calohits from a list of Events objects. Returns the list of
    processed Events. Visualize just the first event

    Parameters
    ----------
        slicer: Slicer object
    
    Returns
    -------
        The list of subtracted Events.
    """
    progbar = Progbar(len(events))
    for i, event in enumerate(events):
        if i == 0 and visualize:
            _, actor_scores = slicer(event, visualize)
            progbar.update(i+1)
            continue
        slicer(event)        
        progbar.update(i+1)
    if visualize:
        print("[+] Rendering inference event")
        start = tm()
        render(events[0], actor_scores, gifname)
        print(f"[+] Saving gif to {gifname}")
        print(f"done, took {tm()-start} s")
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

# TODO: fix plot_plane_view when loaded from results (it always needs the events)
# TODO: fix EMD function in this version