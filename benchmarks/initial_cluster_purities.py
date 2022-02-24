"""
    This module evaluates the purities of the initial clusters. This is meant to
    justify the agglomerative approach of the CM-Net: merging two clusters with
    initial purities p1 and p2, always results in a cluster with a final purity
    pfin lower than p1 and p2.

    Usage:
        python benchmarks/initial_cluster_purities.py <runcard.yaml>
    
    The runcard.yaml should contain the list of archives containing the
"""
import argparse
from pathlib import Path
from time import time as tm
import numpy as np
import matplotlib.pyplot as plt
from slicerl.utils.utils import load_runcard
from slicerl.dataloading.read_data import load_Events_from_files

# ======================================================================
def get_plane_purities(plane, min_cluster_hits=1):
    """
    Computes the cluster purities from plane view. For all initial cluster idx,
    computes its purity, defined as the maximum fraction of hits belonging to
    the same pfo.

    Parameters
    ----------
        - plane: PlaneView, the input plane view
        - min_cluster_hits: int, filters away clusters of reduced size. Default
                            is 1: no filtering.

    Returns
    -------
        - np.array, plane cluster purities
    """
    plane_purities = []
    cluster_idx = list(set(plane.ordered_cluster_idx))

    for cluster_idx in cluster_idx:
        cluster = plane.ordered_cluster_idx == cluster_idx
        nb_cluster_hits = np.count_nonzero(cluster)
        if nb_cluster_hits < min_cluster_hits:
            continue
        cluster_pfos = plane.ordered_mc_idx[cluster]
        pfos_idx = set(cluster_pfos)
        nb_pfos_hits = map(lambda x: np.count_nonzero(cluster_pfos == x), pfos_idx)
        max_hits = sorted(nb_pfos_hits, reverse=True)[0]
        plane_purities.append(max_hits / nb_cluster_hits)
    return np.array(plane_purities)


# ======================================================================
def get_purities(events, min_cluster_hits=1):
    """
    Computes the cluster purities from a list of events.

    Parameters
    ----------
        - events: list, the input events
        - min_cluster_hits: int, filters away clusters of reduced size. Default
                            is 1: no filtering.

    Returns
    -------
        - np.array, cluster purities
    """
    purities = []
    for event in events:
        for plane in event.planes:
            purities.append(get_plane_purities(plane, min_cluster_hits))
    return np.concatenate(purities)


# ======================================================================
def main(runcard, force):
    """
    Benchmark main function.

    Parameters
    ----------
        - runcard: Path, the input runcard
        - force: bool, wether to force purity computation
    """
    fname = Path("../output/proc/purities.npy")

    # compute / load purities
    if fname.is_file() and not force:
        purities = np.load(fname)
    else:
        purities = []
        params = load_runcard(runcard)
        events = load_Events_from_files(params["train"]["fn"], nev=-1)
        purities = get_purities(events)
        np.save(fname, purities)

    # plot purities histogram
    nbins = 100
    binw = 1 / nbins
    bins = np.linspace(0, 1, nbins + 1)
    h = np.histogram(purities, bins=bins, density=True)[0] * binw

    fig = plt.figure(figsize=[8, 6], dpi=125)
    fig.suptitle(
        "ProtoDUNE-SP simulation preliminary: Pandora 2D initial clusters purity",
        y=0.96,
        fontsize=16,
    )
    ax = fig.add_subplot()
    msg = f"Purities mean value: {purities.mean():.5f} +/- {purities.std():.5f}"
    plt.title(msg, fontsize=16)
    ax.hist(
        bins[:-1],
        bins,
        weights=h,
        histtype="step",
        lw=1,
    )
    ax.set_xlabel("Purity", fontsize=16)
    ax.set_ylabel("Fraction of clusters", fontsize=16)
    ax.set_yscale("log")
    plt.savefig("../output/proc/plots/purities_hist.png", bbox_inches="tight")
    plt.close()

    print(msg)


if __name__ == "__main__":
    start = tm()
    parser = argparse.ArgumentParser()
    parser.add_argument("runcard", type=Path, help="runcard file")
    parser.add_argument(
        "--force", action="store_true", help="force purities computation"
    )
    args = parser.parse_args()
    main(args.runcard, args.force)
    print(f"Program done in {tm()-start}s")
