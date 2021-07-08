# This file is part of SliceRL by M. Rossi
from slicerl.config import NP_DTYPE_INT

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

# ----------------------------------------------------------------------
# available cmaps
"""
Color list. An exhaustive list of colors can be retrieved from matplotlib
printing matplotlib.colors.CSS4_COLORS.keys().
"""

colors = [
    "black",
    "deepskyblue",
    "peru",
    "darkorchid",
    "darkgoldenrod",
    "teal",
    "dodgerblue",
    "brown",
    "darkslategrey",
    "turquoise",
    "lightsalmon",
    "plum",
    "darkcyan",
    "orange",
    "slategrey",
    "darkmagenta",
    "limegreen",
    "deeppink",
    "gold",
    "springgreen",
    "midnightblue",
    "green",
    "mediumpurple",
    "mediumvioletred",
    "dimgrey",
    "blueviolet",
    "lightskyblue",
    "darksalmon",
    "royalblue",
    "fuchsia",
    "mediumaquamarine",
    "mediumblue",
    "grey",
    "sienna",
    "mediumslateblue",
    "seagreen",
    "purple",
    "greenyellow",
    "darkviolet",
    "coral",
    "darkblue",
    "goldenrod",
    "lime",
    "cornflowerblue",
    "darkturquoise",
    "orangered",
    "cadetblue",
    "lightcoral",
    "skyblue",
    "mediumseagreen",
    "tomato",
    "blue",
    "pink",
    "olivedrab",
    "rosybrown",
    "darkseagreen",
    "orchid",
    "olive",
    "lightseagreen",
    "cyan",
    "dimgrey",
    "magenta",
    "darkolivegreen",
    "slateblue",
    "lightgreen",
    "navy",
    "indianred",
    "lawngreen",
    "sandybrown",
    "steelblue",
    "salmon",
    "hotpink",
    "darkgrey",
    "violet",
    "cornflowerblue",
    "snow",
    "peru",
    "dimgray",
    "lightyellow",
    "indianred",
    "palegoldenrod",
    "darkgrey",
    "mediumblue",
    "peachpuff",
    "hotpink",
    "green",
    "brown",
    "lightgoldenrodyellow",
    "mediumturquoise",
    "lightslategrey",
    "slateblue",
    "purple",
    "lemonchiffon",
    "orchid",
    "darkred",
    "chocolate",
    "aquamarine",
    "cadetblue",
    "thistle",
    "orange",
    "darkkhaki",
    "yellowgreen",
    "lightsalmon",
    "lightsteelblue",
    "olivedrab",
    "mediumorchid",
    "papayawhip",
    "lime",
    "gainsboro",
    "teal",
    "coral",
    "lightslategray",
    "cyan",
    "lightgrey",
    "honeydew",
    "mediumvioletred",
    "chartreuse",
    "slategray",
    "steelblue",
    "gray",
    "orangered",
    "mediumseagreen",
    "aqua",
    "rebeccapurple",
    "saddlebrown",
    "lawngreen",
    "powderblue",
    "darkseagreen",
    "red",
]
cmap = mpl.colors.ListedColormap(colors)
boundaries = np.arange(len(colors) + 1) - 1.5
norm = mpl.colors.BoundaryNorm(boundaries, cmap.N, clip=True)

l = len(colors)

vcmap = "plasma"
vnorm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

# ----------------------------------------------------------------------
def print_stats(name, data, mass_ref, output_folder="./"):
    """Print statistics on the mass distribution."""
    r_plain = np.array(data) - mass_ref
    m = np.median(r_plain)
    a = np.mean(r_plain)
    s = np.std(r_plain)
    fname = f"{output_folder}/diagnostics.txt"
    with open(fname, "a+") as f:
        print(
            f"{name:<25}:\tmedian-diff {m:.2f}\tavg-diff {a:.2f}\tstd-diff {s:.2f}",
            file=f,
        )


# ----------------------------------------------------------------------
def plot_multiplicity(events, output_folder="./"):
    """Plot the slice multiplicity distribution and output some statistics."""
    nmc = np.array([len(set(event.mc_idx)) for event in events])
    npred = np.array([len(set(event.status)) for event in events])

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
    print(f"[+] Saving plot at {fname} ")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------
def plot_slice_size(events, output_folder="./"):
    """Plot the slice size distribution and output some statistics."""
    bins = np.linspace(50, 1500, 101)
    use_bins = [np.array([0]), bins, np.array([np.inf])]
    use_bins = np.concatenate(use_bins)

    binc_mc = [
        np.bincount(event.ordered_mc_idx.astype(np.int32)) for event in events
    ]
    binc_pred = [
        np.bincount(event.status.astype(np.int32)) for event in events
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
    ax.hist(
        bins[:-1], bins, weights=smc, histtype="step", color="blue", label="mc"
    )
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
    print(f"[+] Saving plot at {fname} ")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------
def plot_plane_view(pc, pc_pred, pc_test, nb_event, output_folder="./"):
    fig = plt.figure(figsize=(18 * 2, 14))
    ax = fig.add_subplot(121)
    ax.scatter(pc[:, 0], pc[:, 1], s=0.5, c=pc_pred, cmap=cmap, norm=norm)
    ax.set_title("pc_pred")

    ax = fig.add_subplot(122)
    ax.scatter(pc[:, 0], pc[:, 1], s=0.5, c=pc_test, cmap=cmap, norm=norm)
    ax.set_title("pc_true")

    fname = f"{output_folder}/pview_{nb_event}.png"
    print(f"[+] Saving plot at {fname} ")
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()


# ----------------------------------------------------------------------
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
    print(f"[+] Saving plot at {fname} ")
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()


# ----------------------------------------------------------------------
def plot_graph(pc, graph, status, output_folder="./"):
    """
    Plots the graph.

    Parameters
    ----------
        - pc     : np.array, point cloud space hits of shape=(N,2)
        - status : np.array, slice index of shape=(N,)
        - slices : list, of sets of connected slices
    """
    fig = plt.figure(figsize=(18, 14))
    ax = fig.add_subplot()

    for node, neighs in enumerate(graph):
        if node in neighs:
            neighs.remove(node)
        if neighs:
            center = np.repeat(pc[node][None], len(neighs), axis=0)
            neighbors = pc[list(neighs)]
            x = np.stack([center[:, 0], neighbors[:, 0]], axis=1).T
            y = np.stack([center[:, 1], neighbors[:, 1]], axis=1).T
            ax.plot(x, y, lw=0.2, color="grey")

    ax.scatter(pc[:, 0], pc[:, 1], s=0.5, c=status, cmap=cmap, norm=norm)

    fname = f"{output_folder}/pred_graph.png"
    print(f"[+] Saving plot at {fname}")
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()
