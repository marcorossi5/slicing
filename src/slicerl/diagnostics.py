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
def match(events, sb_events):
    """ Match LV jets with closest subtracted(LV+PU) jets in (y,phi) plane"""
    events_idxs = []
    for jets_LV, jets_sb in zip(events, sb_events):
        idxs = []
        for jet_noPU in jets_LV:
            min_dist = 100000
            idx = -1
            for i, jet in enumerate(jets_sb):
                dphi = abs(jet_noPU.phi() - jet.phi())
                if (dphi > math.pi):
                    dphi = 2*math.pi - dphi
                drap = jet_noPU.rap()- jet.rap()
                dist = np.sqrt(drap**2 + dphi**2)
                if (dist < min_dist):
                    min_dist = dist
                    idx = i
            idxs.append(idx)
        events_idxs.append(idxs)
    return events_idxs

#----------------------------------------------------------------------
def plot_multiplicity(events, sb_events, output_folder='./', loaddir=None):
    """Plot the jet multiplicity distribution and output some statistics."""
    if loaddir is not None:
        fname = '%s/nnoPU.npy' % loaddir
        nnoPU = np.load(fname)
        fname = '%s/ndqn.npy' % loaddir
        ndqn = np.load(fname)
    else:
        nnoPU = np.array( [len(event) for event in events] )
        ndqn = np.array( [len(event) for event in sb_events] )
    
    bins = np.linspace(-100, 100, 201)
    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(18,14))
    counts, _, _ = plt.hist(ndqn-nnoPU, bins=bins, alpha=0.5,
             linestyle='dotted', facecolor='lawngreen', label='DQN-Subtracting')
    plt.hist(bins[:-1], bins, weights=counts, histtype='step', color='green', linestyle='dotted')

    plt.xlabel("$n_{jets,reco}- n_{jets,LV}$", loc='right')
    plt.xlim((bins[0], bins[-1]))
    plt.legend()
    fname = f"{output_folder}/multiplicity.pdf"
    plt.savefig(fname, bbox_inches='tight')
        
    print_stats('jet multiplicity', ndqn  , nnoPU, output_folder=output_folder)
    resultsdir = '%s/results' % output_folder
    np.save(f"{resultsdir}/nnoPU.npy", nnoPU)
    np.save(f"{resultsdir}/ndqn.npy", ndqn)

#----------------------------------------------------------------------
def plot_mass(events, sb_events, output_folder='./', loaddir=None):
    """Plot the mass distribution and output some statistics."""
    if loaddir is not None:
        fname = '%s/mnoPU.npy' % loaddir
        mnoPU = np.load(fname)
        fname = '%s/mdqn.npy' % loaddir
        mdqn = np.load(fname)
    else:
        mnoPU = np.array( [jet.m() for event in events for jet in event] )
        mdqn = np.array( [jet.m() for event in sb_events for jet in event] )
    
    bins = np.linspace(-2.5, 2.5, 101)
    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(18,14))
    counts, _, _ = plt.hist((mdqn-mnoPU)/mnoPU, bins=bins, alpha=0.5,
             linestyle='dotted', facecolor='lawngreen', label='DQN-Subtracting')
    plt.hist(bins[:-1], bins, weights=counts, histtype='step', color='green', linestyle='dotted')
    # plt.hist((mplain-mnoPU)/mnoPU, bins=bins, histtype='step', color='blue', lw=1, label='plain')

    plt.xlabel("$(m_{reco}- m_{LV})/m_{LV}$", loc='right')
    plt.xlim((bins[0], bins[-1]))
    plt.legend()
    fname = f"{output_folder}/mass.pdf"
    plt.savefig(fname, bbox_inches='tight')
        
    # print_stats('mplain  ', mplain, mnoPU, output_folder=output_folder)
    print_stats('mdqn    ', mdqn  , mnoPU, output_folder=output_folder)
    resultsdir = '%s/results' % output_folder
    np.save(f"{resultsdir}/mnoPU.npy", mnoPU)
    np.save(f"{resultsdir}/mdqn.npy", mdqn)

#----------------------------------------------------------------------
def plot_pT(events, sb_events, output_folder='./', loaddir=None):
    """Plot the mass distribution and output some statistics."""
    if loaddir is not None:
        fname = '%s/pTnoPU.npy' % loaddir
        pTnoPU = np.load(fname)
        fname = '%s/pTdqn.npy' % loaddir
        pTdqn = np.load(fname)
    else:
        pTnoPU = np.array( [np.sqrt(jet.px()**2 + jet.py()**2) for event in events for jet in event] )
        pTdqn = np.array( [np.sqrt(jet.px()**2 + jet.py()**2) for event in sb_events for jet in event] )
    
    bins = np.linspace(-2, 10, 51)
    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(18,14))
    counts, _, _ = plt.hist((pTdqn-pTnoPU)/pTnoPU, bins=bins, alpha=0.5,
             linestyle='dotted', facecolor='lawngreen', label='DQN-Subtracting')
    plt.hist(bins[:-1], bins, weights=counts, histtype='step', color='green', linestyle='dotted')
    # plt.hist((pTplain-pTnoPU)/pTnoPU, bins=bins, histtype='step', color='blue', lw=1, label='plain')

    plt.xlabel("$(p_{T,reco}- p_{T,LV})/p_{T,LV}$", loc='right')
    plt.xlim((bins[0], bins[-1]))
    plt.legend()
    fname = f"{output_folder}/pT.pdf"
    plt.savefig(fname, bbox_inches='tight')
        
    # print_stats('pTplain', pTplain, pTnoPU, output_folder=output_folder)
    print_stats('pTdqn'  , pTdqn  , pTnoPU, output_folder=output_folder)
    resultsdir = '%s/results' % output_folder
    np.save(f"{resultsdir}/pTnoPU.npy", pTnoPU)
    np.save(f"{resultsdir}/pTdqn.npy", pTdqn)

#----------------------------------------------------------------------
def plot_EMD(events, sb_events, output_folder='./', loaddir=None):
    """Plot the Energy Moving Distance distribution and output some statistics."""
    if loaddir is not None:
        fname = '%s/EMDdqn.npy' % loaddir
        EMDdqn = np.load(fname)
    else:
        EMDdqn = np.array([jet_emd(jet_noPU, sb_jet) for jets_noPU, sb_jets in zip(events, sb_events) \
                                                for jet_noPU, sb_jet in zip(jets_noPU, sb_jets)])
    bins = np.arange(0, 300, 2)
    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(18,14))
    counts, _, _ = plt.hist(EMDdqn, bins=bins, alpha=0.5,
             linestyle='dotted', facecolor='lawngreen', label='DQN-Subtracting')
    plt.hist(bins[:-1], bins, weights=counts, histtype='step', color='green', linestyle='dotted')
    # plt.hist(EMDplain, bins=bins, histtype='step', color='blue', lw=1, label='plain')

    plt.xlabel("$EMD_{reco}$", loc='right')
    plt.xlim((bins[0], bins[-1]))
    plt.legend()
    fname = f"{output_folder}/EMD.pdf"
    plt.savefig(fname, bbox_inches='tight')
        
    print_stats('EMDdqn'  , EMDdqn  , 0, output_folder=output_folder)
    resultsdir = '%s/results' % output_folder
    np.save(f"{resultsdir}/EMDdqn.npy", EMDdqn)

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
def inference(subtractor, events):
    """
    Subtract PU particles from a list of Events objects. Returns a list of
    subtracted Events.

    Parameters
    ----------
        subtractor: Subtractor object
    
    Returns
    -------
        The list of subtracted Events.
    """
    events_sb = []
    progbar = Progbar(len(events))
    for i, event in enumerate(events):
        events_sb.append( subtractor(event) )
        progbar.update(i+1)
    return events_sb

#----------------------------------------------------------------------
def make_plots(events, jet_algorithm, R, plotdir):
    """
    Make diagnostics plots from a list of subtracted events.

    Parameters
    ----------
        events: list of Event objects
            list of subtracted events
        jet_algorithm: fastjet::JetAlgorithm
            the jet algorithm to re-cluster events for PU subtraction assessment
        - R: float
            radius parameter of jet clustering algorithm
        - plotdir: str
            plots output folder
    """
    events_LV               = []
    events_reco             = []
    kept_rejected_particles = []
    # collects (truth, prediction) for particle to be PU: 1 is PU, 0 is LV
    
    jet_def = fj.JetDefinition(jet_algorithm, R) 
    
    for event in events:
        ppjets = event.particles_as_pseudojets()

        p_LV    = []
        p_reco  = []
        krp     = []

        for p in ppjets:
            assert p.has_user_info(), "Particles do not carry user info, ensure to set load_truth to True in Reader object"
            PU = p.python_info().PU
            status = p.python_info().status
            if PU == 0:
                p_LV.append( p )
            if status == 1:
                p_reco.append( p )
            krp.append( [PU, 1-status] )

        events_LV.append( jet_def(p_LV) )
        events_reco.append( jet_def(p_reco) )
        kept_rejected_particles.append( np.array(krp) )
    
    events_idxs = match(events_LV, events_reco)

    # sort the subtracted jets according to closest ones
    # drop the unmatched ones
    events_sorted = [ [jets_sb[idx] for idx in jets_idx] for jets_idx, jets_sb in zip(events_idxs, events_reco) ]

    plot_multiplicity(events_LV, events_reco, plotdir)
    plot_mass(events_LV, events_sorted, plotdir)
    plot_pT(events_LV, events_sorted, plotdir)
    plot_EMD(events_LV, events_sorted, plotdir)
    plot_ROC(kept_rejected_particles, plotdir)

#----------------------------------------------------------------------
def load_and_dump_plots(plotdir, loaddir):
    """
    Make diagnostics plots from plot data contained in loaddir.

    Parameters
    ----------
        - plotdir: str
            plots output folder
        - loaddir: str
            directory where to load plot data from
    """
    plot_multiplicity(None, None, plotdir, loaddir)
    plot_mass(None, None, plotdir, loaddir)
    plot_pT(None, None, plotdir, loaddir)
    plot_EMD(None, None, plotdir, loaddir)
    plot_ROC(None, plotdir, loaddir)