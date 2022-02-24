import logging
import numpy as np
import matplotlib.pyplot as plt
from slicerl import PACKAGE

logger = logging.getLogger(PACKAGE + ".diagnostics")

THRESHOLD = 0.9  # the purity / completeness threshold


def get_beam_metrics(events, pndr=False, dump=False):
    """Returns the beam metrics."""
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
                    f"Test beam slice {islice+1}/{len(beam_mc_slices)}, "
                    f"mc Hits: {tot_this_mc_TB}:"
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
                    print_str = (
                        "  Is{result},"
                        + f" reco Hits: {nb_pred_hits[-1]},  "
                        + f"Purity: {purities[-1]*100:.2f}%, "
                        + f"Completeness {completenesses[-1]*100:.2f}%"
                    )
                    if purity >= THRESHOLD:
                        logger.debug(print_str.format(result="Correct"))
                        statuses.append([1, 0, 0])
                    else:
                        logger.debug(print_str.format(result="Lost"))
                        statuses.append([0, 1, 0])
                elif nb_p_slices > 1:
                    cs = completenesses[-nb_p_slices:]
                    amax_cs = np.argmax(cs)
                    if np.max(cs) > THRESHOLD:
                        # check the purity of the most complete slice
                        nb = nb_pred_hits[-nb_p_slices:][amax_cs]
                        print_str = (
                            f"  Is(result), reco Hits: {nb},  "
                            + f"Purity: {purities[-nb_p_slices:][amax_cs]*100:.2f}%, "
                            + f"Completeness {completenesses[-nb_p_slices:][amax_cs]*100:.2f}%"
                        )
                        p = purities[-nb_p_slices:][amax_cs]
                        if p > THRESHOLD:
                            logger.debug(print_str.format(result="Correct"))
                            this_status = [1, 0, 0]
                        else:
                            logger.debug(print_str.format(result="Lost"))
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
    fname = output_folder / "TB_purity_completeness.png"
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
    fname = output_folder / "TB_purity_completeness_heatmap.png"
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
        label="Network",
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
    fname = output_folder / "TB_purity.png"
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
        label="Network",
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
    fname = output_folder / "TB_completeness.png"
    logger.info(f"Saving plot at {fname} ")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()


# ======================================================================
def plot_test_beam_metrics(events, output_folder="./"):
    """Plot the test beam metrics distribution and output some statistics."""
    beam_metrics = get_beam_metrics(events)

    logger.info(
        "Evaluating metrics Cluster Merging Network ...\n"
        + print_beam_metrics(beam_metrics)
    )

    beam_pndr_metrics = get_beam_metrics(events, pndr=True)
    logger.info(
        "Evaluating metrics Pandora ...\n" + print_beam_metrics(beam_pndr_metrics)
    )

    plot_purity_completeness(beam_metrics, beam_pndr_metrics, output_folder)


# ======================================================================
def check_beam_metrics(events):
    all_statuses = []  # correct, lost, split status
    all_purities = []
    all_completenesses = []
    all_nb_pred_hits = []
    all_nb_mc_hits = []

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


# ======================================================================
def print_beam_metrics(beam_metrics):
    """Returns the formatted message with the beam metrics to be printed."""
    m = beam_metrics[0].sum(0) > 0
    nb_tests = np.count_nonzero(m)
    stats = beam_metrics[0][:, m]
    return (
        "------------------------------\n"
        "- Including all slices\n"
        f"  isCorrect: {stats[0].sum()}/{nb_tests}, {stats[0].sum()/nb_tests*100:.2f}%\n"
        f"  isLost: {stats[1].sum()}/{nb_tests}, {stats[1].sum()/nb_tests*100:.2f}%\n"
        f"  isSplit: {stats[2].sum()}/{nb_tests}, {stats[2].sum()/nb_tests*100:.2f}%"
    )
