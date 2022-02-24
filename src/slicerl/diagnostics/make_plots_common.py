import logging
import numpy as np
import matplotlib.pyplot as plt
from .test_beam_plots import plot_test_beam_metrics
from slicerl import PACKAGE

logger = logging.getLogger(PACKAGE + ".diagnostics")

def make_plots(generator, folder):
    plot_test_beam_metrics(generator.events, folder)
    plot_slice_size(generator.events, folder)
    plot_multiplicity(generator.events, folder)


# ======================================================================
def plot_histogram(y_true, y_pred, output_folder="./"):
    """
    Parameters
    ----------
        - y_true : list, predictions each of shape=(num hits*(1+K))
        - y_pred : list, predictions each of shape=(num hits*(1+K))
    """
    bins = np.linspace(0, 1, 201)
    plt.rcParams.update(
        {
            "ytick.labelsize": 13,
            "xtick.labelsize": 13,
            "axes.labelsize": 16,
            "axes.titlesize": 28,
            "legend.fontsize": 13,
        }
    )

    h_edged = []
    h_non_edged = []
    for truths, preds in zip(y_true, y_pred):
        mask = truths == 1
        pred_edged = preds[mask]
        pred_non_edged = preds[~mask]

        h_edged.append(np.histogram(pred_edged, bins=bins)[0])
        h_non_edged.append(np.histogram(pred_non_edged, bins=bins)[0])

    mean_edged = np.array(h_edged).mean(0)
    std_edged = np.array(h_edged).std(0)

    mean_non_edged = np.array(h_non_edged).mean(0)
    std_non_edged = np.array(h_non_edged).std(0)

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.hist(
        bins[:-1],
        bins,
        weights=mean_edged,
        histtype="step",
        lw=0.7,
        color="green",
        label="Positives",
    )
    ax.fill_between(
        bins[:-1],
        mean_edged - std_edged,
        mean_edged + std_edged,
        color="green",
        alpha=0.4,
        step="post",
        edgecolor=None,
    )
    ax.hist(
        bins[:-1],
        bins,
        weights=mean_non_edged,
        histtype="step",
        lw=0.7,
        color="red",
        label="Negatives",
    )
    ax.fill_between(
        bins[:-1],
        mean_non_edged - std_non_edged,
        mean_non_edged + std_non_edged,
        color="red",
        alpha=0.4,
        step="post",
        edgecolor=None,
    )

    ax.set_yscale("log")
    ax.legend()
    fname = f"{output_folder}/pred_hist.png"
    logger.info(f"Saving plot at {fname} ")
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()


# ======================================================================
def plot_slice_size(events, output_folder="./"):
    """Plot the slice size distribution and output some statistics."""
    bins = np.linspace(50, 1500, 101)
    use_bins = [np.array([0]), bins, np.array([np.inf])]
    use_bins = np.concatenate(use_bins)

    binc_mc = [
        np.bincount(plane.ordered_mc_idx.astype(np.int32))
        for event in events
        for plane in event.planes
    ]
    binc_pred = [
        np.bincount(plane.status.astype(np.int32))
        for event in events
        for plane in event.planes
    ]
    smc = sum([np.histogram(bc, bins=use_bins)[0] for bc in binc_mc])
    spred = sum([np.histogram(bc, bins=use_bins)[0] for bc in binc_pred])

    uflow_mc = smc[0]
    oflow_mc = smc[1]

    uflow_pred = spred[0]
    oflow_pred = spred[1]

    smc = smc[1:-1]
    spred = spred[1:-1]

    plt.rcParams.update({"font.size": 20})
    fig = plt.figure(figsize=(9, 7))

    ax = fig.add_subplot()
    ax.hist(bins[:-1], bins, weights=smc, histtype="step", color="blue", label="mc")
    ax.hist(
        bins[:-1],
        bins,
        weights=spred,
        histtype="step",
        color="red",
        label="net",
    )

    ax.set_xlabel("size", loc="right")
    ax.set_xlim((bins[0], bins[-1]))

    textstr = f"Underflow   Overflow\nmc{uflow_mc:>7}  {oflow_mc:>9}\nnet{uflow_pred:>6}  {oflow_pred:>9}"

    props = dict(boxstyle="round", facecolor="white", alpha=0.5)

    ax.text(
        0.68,
        0.78,
        textstr,
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )

    ax.legend()

    fname = f"{output_folder}/slice_size.png"
    logger.info(f"Saving plot at {fname} ")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()


# ======================================================================
def plot_multiplicity(events, output_folder="./"):
    """Plot the slice multiplicity distribution and output some statistics."""
    nmc = np.array(
        [len(set(plane.mc_idx)) for event in events for plane in event.planes]
    )
    npred = np.array(
        [len(set(plane.status)) for event in events for plane in event.planes]
    )

    bins = np.linspace(0, 127, 128)
    hnmc, _ = np.histogram(nmc, bins=bins)
    hnpred, _ = np.histogram(npred, bins=bins)

    plt.rcParams.update({"font.size": 20})
    plt.figure(figsize=(9, 7))

    plt.hist(
        bins[:-1],
        bins,
        weights=hnmc,
        histtype="step",
        color="blue",
        label="mc",
    )
    plt.hist(
        bins[:-1],
        bins,
        weights=hnpred,
        histtype="step",
        color="red",
        label="net",
    )

    plt.xlabel("multiplicity", loc="right")
    plt.xlim((bins[0], bins[-1]))
    plt.legend()
    fname = f"{output_folder}/multiplicity.png"
    logger.info(f"Saving plot at {fname} ")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
