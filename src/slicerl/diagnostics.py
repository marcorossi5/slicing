# This file is part of SliceRL by M. Rossi
import logging
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from slicerl import PACKAGE

logger = logging.getLogger(PACKAGE + ".diagnostics")

THRESHOLD = 0.9


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

# ======================================================================
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
def get_beam_metrics(events, pndr=False, dump=False):

    statuses = []  # correct, lost, split status
    purities = []
    completenesses = []
    nb_pred_hits = []
    nb_mc_hits = []

    for iev, event in enumerate(events):
        for plane in event.planes:
            if pndr:
                status = plane.ordered_pndr_idx
            else:
                status = plane.status

            isBeam = plane.test_beam.astype(bool)
            tot_TB = np.count_nonzero(isBeam)
            mc_beam = plane.ordered_mc_idx[isBeam]
            beam_mc_slices = list(set(mc_beam))

            logger.debug("------------------------------")
            logger.debug(f"plane {iev}:")

            for islice, idx in enumerate(beam_mc_slices):
                slice_mc = (
                    plane.ordered_mc_idx == idx
                )  # hyp: all of these are test beam

                tot_this_mc_TB = np.count_nonzero(
                    slice_mc
                )  # total hits for this mc slice
                logger.debug(
                    f"Test beam slice {islice+1}/{len(beam_mc_slices)}, mc Hits: {tot_this_mc_TB}:"
                )

                pred_beam = status[slice_mc]
                beam_pred_slices = list(set(pred_beam))
                nb_p_slices = len(beam_pred_slices)

                for p_idx in beam_pred_slices:
                    # compute purity and completeness
                    # purity is percentage of test beam hits in reconstructed slice
                    # completeness is fraction of test beam hits in reconstructed
                    # slice over the total number of test beam hits in the plane
                    sl = status == p_idx

                    tot = np.count_nonzero(sl)  # total hits in reco slice
                    isBeam_purity = np.count_nonzero(
                        plane.test_beam[sl]
                    )  # total hits in slice that are TB
                    purity = isBeam_purity / tot
                    # isBeam_completeness = np.count_nonzero(
                    #     np.logical_and(slice_mc, sl)
                    # )
                    # completeness = isBeam_completeness / tot_mc
                    # completeness = isBeam_purity / tot_TB
                    c_num = np.count_nonzero(np.logical_and(slice_mc, sl))
                    completeness = np.clip(c_num, None, tot_this_mc_TB) / tot_this_mc_TB

                    nb_mc_hits.append(tot_this_mc_TB)
                    nb_pred_hits.append(tot)
                    purities.append(purity)
                    completenesses.append(completeness)

                if nb_p_slices == 1:
                    if purity >= THRESHOLD:
                        print_str = f"  Is%(result)s, reco Hits: {nb_pred_hits[-1]},  Purity: {purities[-1]*100:.2f}%, Completeness {completenesses[-1]*100:.2f}%"
                        logger.debug(print_str % {"result": "Correct"})
                        statuses.append([1, 0, 0])
                    else:
                        logger.debug(print_str % {"result": "Lost"})
                        statuses.append([0, 1, 0])
                elif nb_p_slices > 1:
                    cs = completenesses[-nb_p_slices:]
                    amax_cs = np.argmax(cs)
                    if np.max(cs) > THRESHOLD:
                        # check the purity of the most complete slice
                        nb = nb_pred_hits[-nb_p_slices:][amax_cs]
                        print_str = f"  Is%(result)s, reco Hits: {nb},  Purity: {purities[-nb_p_slices:][amax_cs]*100:.2f}%, Completeness {completenesses[-nb_p_slices:][amax_cs]*100:.2f}%"
                        p = purities[-nb_p_slices:][amax_cs]
                        if p > THRESHOLD:
                            logger.debug(print_str % {"result": "Correct"})
                            this_status = [1, 0, 0]
                        else:
                            logger.debug(print_str % {"result": "Lost"})
                            this_status = [0, 1, 0]
                    else:
                        logger.debug(f"  IsSplit in {nb_p_slices} slices:")
                        this_status = [0, 0, 1]
                    # multiple_statuses = [[0,0,0] for i in range(amax_cs - 1)] + [this_status] + [[0,0,0] for i in range(amax_cs + 1, nb_p_slices)]
                    # statuses.extend(multiple_statuses) # must keep the number of statuses equal to (reco slices * mc slices)
                    statuses.append(this_status)
                else:
                    raise ValueError(
                        f"nb_p_slices is {nb_pred_hits}, we should not be here !!"
                    )

    statuses = np.array(statuses).T  # of shape=(3, tests)
    purities = np.array(purities)
    completenesses = np.array(completenesses)
    nb_pred_hits = np.array(nb_pred_hits)
    nb_mc_hits = np.array(nb_mc_hits)
    return statuses, purities, completenesses, nb_pred_hits, nb_mc_hits


# ======================================================================
def print_beam_metrics(beam_metrics):
    m = beam_metrics[0].sum(0) > 0
    nb_tests = np.count_nonzero(m)
    # filtered_m = np.logical_and(
    #     beam_metrics[4] > 5, m
    # )  # do not double count the splittings
    # nb_filtered_tests = np.count_nonzero(filtered_m)

    stats = beam_metrics[0][:, m]
    # filtered_stats = beam_metrics[0][:, filtered_m]

    logger.info("------------------------------\n- Including all slices")
    logger.info(
        f"  isCorrect: {stats[0].sum()}/{nb_tests}, {stats[0].sum()/nb_tests*100:.2f}%"
    )
    logger.info(f"  isLost: {stats[1].sum()}/{nb_tests}, {stats[1].sum()/nb_tests*100:.2f}%")
    logger.info(f"  isSplit: {stats[2].sum()}/{nb_tests}, {stats[2].sum()/nb_tests*100:.2f}%")

    # print("- Filtering out small slices (< 5 mc hits)")
    # print(
    #     f"  isCorrect: {filtered_stats[0].sum()}/{nb_filtered_tests}, {filtered_stats[0].sum()/nb_filtered_tests*100:.2f}%"
    # )
    # print(
    #     f"  isLost: {filtered_stats[1].sum()}/{nb_filtered_tests}, {filtered_stats[1].sum()/nb_filtered_tests*100:.2f}%"
    # )
    # print(
    #     f"  isSplit: {filtered_stats[2].sum()}/{nb_filtered_tests}, {filtered_stats[2].sum()/nb_filtered_tests*100:.2f}%"
    # )


# ======================================================================
def plot_purity_completeness(beam_metrics, beam_pndr_metrics, output_folder):
    # purity vs completeness scatterplot
    plt.rcParams.update({"font.size": 20})
    plt.figure(figsize=(9, 7))
    plt.scatter(beam_metrics[2], beam_metrics[1], color="green", s=1, label="CM-Net")
    plt.scatter(
        beam_pndr_metrics[2], beam_pndr_metrics[1], color="red", s=1, label="Pandora"
    )
    plt.xlabel("Completeness")
    plt.ylabel("Purity")
    plt.legend()
    plt.grid(alpha=0.4)
    fname = output_folder / "plots/TB_purity_completeness.png"
    logger.info(f"Saving plot at {fname} ")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()

    # purity vs completeness heatmap
    # Warning: the imshow function plots exchanging the x and y axes
    plt.figure(figsize=(9 * 2, 7))
    cbins = np.linspace(0.75, 1, 26)
    ybins = np.linspace(0, 1, 21)
    tot = len(beam_metrics)
    h_cma, _, _ = np.histogram2d(beam_metrics[1], beam_metrics[2], bins=[cbins, ybins])
    plt.subplot(121)
    plt.title("Network output")
    plt.xlabel("Completeness")
    plt.ylabel("Purity")
    plt.imshow(
        h_cma / tot,
        vmin=0,
        vmax=1,
        origin="lower",
        extent=[cbins[0], cbins[-1], ybins[0], ybins[-1]],
        aspect="auto",
    )
    plt.colorbar()
    h_pndr, _, _ = np.histogram2d(
        beam_pndr_metrics[1], beam_pndr_metrics[2], bins=[cbins, ybins]
    )
    plt.subplot(122)
    plt.title("Pandora output")
    plt.xlabel("Completeness")
    plt.ylabel("Purity")
    plt.imshow(
        h_pndr / tot,
        vmin=0,
        vmax=1,
        origin="lower",
        extent=[cbins[0], cbins[-1], ybins[0], ybins[-1]],
        aspect="auto",
    )
    plt.colorbar()
    fname = output_folder / "plots/TB_purity_completeness_heatmap.png"
    logger.info(f"Saving plot at {fname} ")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()

    # purity histogram
    bins = np.linspace(0, 1, 101)
    h, _ = np.histogram(beam_metrics[1], bins=bins)
    h_pndr, _ = np.histogram(beam_pndr_metrics[1], bins=bins)
    plt.rcParams.update({"font.size": 20})
    plt.figure(figsize=(9, 7))
    plt.title("Slice Purity Distribution")
    plt.hist(
        bins[:-1],
        bins,
        weights=h,
        histtype="step",
        color="green",
        label="CM-Net",
        lw=0.5,
    )
    plt.hist(
        bins[:-1],
        bins,
        weights=h_pndr,
        histtype="step",
        color="red",
        label="Pandora",
        lw=0.5,
    )
    plt.yscale("log")
    plt.xlabel("Purity")
    plt.legend(loc="upper left")
    fname = output_folder / "plots/TB_purity.png"
    logger.info(f"Saving plot at {fname} ")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()

    # completeness histogramm
    h, _ = np.histogram(beam_metrics[2], bins=bins)
    h_pndr, _ = np.histogram(beam_pndr_metrics[2], bins=bins)
    plt.rcParams.update({"font.size": 20})
    plt.figure(figsize=(9, 7))
    plt.title("Slice Completeness Distribution")
    plt.hist(
        bins[:-1],
        bins,
        weights=h,
        histtype="step",
        color="green",
        label="CM-Net",
        lw=0.5,
    )
    plt.hist(
        bins[:-1],
        bins,
        weights=h_pndr,
        histtype="step",
        color="red",
        label="Pandora",
        lw=0.5,
    )
    plt.yscale("log")
    plt.xlabel("Completeness")
    plt.legend(loc="upper left")
    fname = output_folder / "plots/TB_completeness.png"
    logger.info(f"Saving plot at {fname} ")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()


# ======================================================================
def plot_test_beam_metrics(events, output_folder="./"):
    """Plot the test beam metrics distribution and output some statistics."""
    all_statuses = []  # correct, lost, split status
    all_purities = []
    all_completenesses = []
    all_nb_pred_hits = []
    all_nb_mc_hits = []

    beam_metrics = get_beam_metrics(events)
    logger.info("\nEvaluating metrics Cluster Merging Network ...")
    print_beam_metrics(beam_metrics)
    # plot_purity_completeness(beam_metrics, output_folder + "purity_completeness.png")

    beam_pndr_metrics = get_beam_metrics(events, pndr=True)
    logger.info("\nEvaluating metrics Pandora ...")
    print_beam_metrics(beam_pndr_metrics)

    plot_purity_completeness(beam_metrics, beam_pndr_metrics, output_folder)

    return

    for iev, event in enumerate(events):
        logger.info("------------------------------")
        logger.info(f"Event {iev}:")

        # beam_metrics = get_beam_metrics(event.status, event.ordered_mc_idx, event.calohits[-2])
        beam_metrics = get_beam_metrics(
            event.ordered_pndr_idx, event.ordered_mc_idx, event.calohits[-2]
        )

        all_statuses.extend(beam_metrics[0])
        all_purities.extend(beam_metrics[1])
        all_completenesses.extend(beam_metrics[2])
        all_nb_pred_hits.extend(beam_metrics[3])
        all_nb_mc_hits.extend(beam_metrics[4])

    all_statuses = np.array(all_statuses).T  # of shape=(3, all_tests)
    all_purities = np.array(all_purities)
    all_completenesses = np.array(all_completenesses)
    all_nb_pred_hits = np.array(all_nb_pred_hits)
    all_nb_mc_hits = np.array(all_nb_mc_hits)

    m = all_statuses.sum(0) > 0
    nb_tests = np.count_nonzero(m)
    filtered_m = np.logical_and(
        all_nb_mc_hits > 5, m
    )  # do not double count the splittings
    nb_filtered_tests = np.count_nonzero(filtered_m)
    print_beam_metrics(
        all_statuses[:, m],
        nb_tests,
        all_statuses[:, filtered_m],
        nb_filtered_tests,
    )

    exit()


# ======================================================================
def plot_plane_view(
    pc, pc_init, pc_pred, pc_pndr, pc_test, nb_event, output_folder="./"
):
    fig = plt.figure(figsize=(9 * 3, 14))
    ax = fig.add_subplot(221)
    ax.scatter(pc[:, 0], pc[:, 1], s=0.5, c=pc_init, cmap=cmap, norm=norm)
    ax.set_title("pc_init")

    ax = fig.add_subplot(222)
    ax.scatter(pc[:, 0], pc[:, 1], s=0.5, c=pc_pred, cmap=cmap, norm=norm)
    ax.set_title("pc_pred")

    ax = fig.add_subplot(223)
    ax.scatter(pc[:, 0], pc[:, 1], s=0.5, c=pc_pndr, cmap=cmap, norm=norm)
    ax.set_title("pc_pndr")

    ax = fig.add_subplot(224)
    ax.scatter(pc[:, 0], pc[:, 1], s=0.5, c=pc_test, cmap=cmap, norm=norm)
    ax.set_title("pc_true")

    fname = f"{output_folder}/pview_{nb_event}.png"
    logger.info(f"Saving plot at {fname} ")
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()


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
    logger.info(f"Saving plot at {fname}")
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()
