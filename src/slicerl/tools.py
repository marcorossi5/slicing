# This file is part of SliceRL by M. Rossi

import numpy as np
import math
from energyflow.emd import emd

eps = np.finfo(np.float64).eps

#----------------------------------------------------------------------
def mass(events, noPU=False):
    """Given a list of events, determine the masses for each jet inside."""

    masses = []

    for event in events:
        for jet_noPU, jet in event.jet_pairs:
            # masses_per_event = []
            p = jet_noPU if noPU else jet
            msq = p.E**2 - p.px**2 - p.py**2 - p.pz**2
            # masses_per_event.append(math.sqrt(msq) if msq > 0.0 else -math.sqrt(-msq))
            masses.append(math.sqrt(msq) if msq > 0.0 else -math.sqrt(-msq))
        # masses.append(masses_per_event)

    return masses

#----------------------------------------------------------------------
def pT(events, noPU=False):
    """Given a list of events, determine the pT for each jet inside."""

    return  [math.sqrt(p.px()**2 + p.py()**2) for event in events for jet in event]

#----------------------------------------------------------------------
def get_window_width(masses, lower_frac=20, upper_frac=80):
    """Returns"""
    lower = np.nanpercentile(masses, lower_frac)
    upper = np.nanpercentile(masses, upper_frac)
    median = np.median(masses[(masses > lower) & (masses < upper)])
    return lower, upper, median

#----------------------------------------------------------------------
def quality_metric(slice_state, mc_state):
    """ Caluclate EMD metric for two Particle objects """
    return emd(slice_state, mc_state, R=0.6)

#----------------------------------------------------------------------
def jet_emd(j_noPU, j):
    """ Caluclate EMD metric for two FastJet.PseudoJet objects """
    pT = lambda x: math.sqrt(x.px()**2 + x.py()**2)
    ev0 = np.array([pT(j_noPU), j_noPU.rap(), j_noPU.phi()])
    ev1 = np.array([pT(j), j.rap(), j.phi()])
    return emd(ev0, ev1)

#----------------------------------------------------------------------
def confusion_matrix_per_event(scores):
    """
    Computes confusion matrix values for each jet in list
      - scores  list of event numpy array scores per event: first level events, 
                second level arrays of shape [num particles, 2]
    Returns 
      - tp      true positives ratio in jet distribution
      - fp      false positives ratio in jet distribution
      - fn      false negatives ratio in jet distribution
      - tn      true negatives ratio in jet distribution
    """
    tp = []
    fp = []
    fn = []
    tn = []
    for score in scores:
        if score.shape[0] == 0:
            continue
        truths = score[:,0].astype(bool)
        preds  = score[:,1]
        tot = score.shape[0]
        tp.append(len(np.where(preds[truths]  == 1)[0])/tot)
        fp.append(len(np.where(preds[~truths] == 1)[0])/tot)
        fn.append(len(np.where(preds[truths]  == 0)[0])/tot)
        tn.append(len(np.where(preds[~truths] == 0)[0])/tot)
    
    return np.array(tp), np.array(fp), np.array(fn), np.array(tn)

#----------------------------------------------------------------------
def m_lin_fit(x,y):
    """ Compute the angular coefficient of a linear fit. """
    assert x.shape == y.shape
    n = x.shape[0]
    num = n * (x*y).sum() - x.sum()*y.sum()
    den = n * (x*x).sum() - (x.sum())**2
    return num / (den + eps)

#----------------------------------------------------------------------
def pearson_distance(x,y):
    """ Computes modified pearson distance. """
    xc = x - x.mean()
    yc = y - y.mean()
    num = (xc * yc).sum()
    den = (xc**2).sum() * (yc**2).sum()
    return 1 - num**2 / (den + eps)

#----------------------------------------------------------------------
def mse(x,y):
    return ((x-y)**2).mean()

#----------------------------------------------------------------------
def bce_loss(x,y):
    # Warning: computing log is expensive
    ratio = 0.1 # percentage of ones over zeros
    loss = - y*np.log(x + eps)/ratio - (1-y)*np.log(1-x + eps)/(1-ratio)
    return loss.mean()

#----------------------------------------------------------------------
def dice_loss(x,y):
    ix = 1-x
    iy = 1-y
    num1 = (x*y).sum(-1) + eps
    den1 = (x*x + y*y).sum(-1) + eps
    num2 = (ix*iy).sum(-1) + eps
    den2 = (ix*ix + iy*iy).sum(-1) + eps
    return 1 - (num1/den1 + num2/den2).mean()

#----------------------------------------------------------------------
def efficiency_rejection_rate_loss(x,y):
    efficiency     = np.count_nonzero(x[y]) / (np.count_nonzero(y) + eps )
    rejection_rate = np.count_nonzero(x[~y]) / (np.count_nonzero(~y) + eps)
    return 1 - efficiency + rejection_rate