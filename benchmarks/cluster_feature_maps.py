"""
    This script loads a CM-Net from output folder and plots the intermediate
    cluster feature maps. It produces the
    `cluster_features<start feature>-<end feature>.png` in the `<output>/plot`
    folder.

    Usage (from package root directory):

    ```
    python benchmarks/cluster_feature_maps.py
    ```
"""
import argparse
from pathlib import Path
from time import time as tm
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from slicerl.utils.config import preconfig_tf
from slicerl.utils.utils import load_runcard, modify_runcard
from slicerl.build_dataset import build_dataset_from_np
from slicerl.build_model import load_network

# from slicerl.CMNet import ReduceMax

from sklearn.feature_selection import mutual_info_classif
import seaborn as sns
from scipy.stats import gaussian_kde


def plot_clusters(inputs, y_pred, y_true, fidx, fname):
    """
    Parameters
    ----------
        - inputs: np.array, inputs of length `events`. Each element of
                            shape=([nb hits], nb feats)
        - y_pred: np.array, extracted features of length `events`. Each element
                            of shape=([nb hits], nb feats)
        - y_true: np.array, target labels of shape=(events,)
        - fidx: int, feature idx to plot
        - fname: Path, the output plot file name
    """
    nrow = 4
    ncol = 4
    fig = plt.figure(figsize=[6.4 * nrow, 4.8 * ncol])
    axs = fig.subplots(nrow, ncol)

    for r in range(nrow):
        for c in range(ncol):
            n = r * ncol + c
            axs[r, c].scatter(inputs[n][:, 1]*1000, inputs[n][:, 2]*1000, c=y_pred[n][:,fidx], s=1)
            axs[r, c].title.set_text(f"Should merge? {'Yes' if y_true[n] else 'No'}")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()


def plot_features(feats, y_true, fname, mis=None, t=0.2, ffeat=0):
    """
    Plots the extracted features.

    Parameters
    ----------
        - feats: np.array, features array of shape=(events, nb feats)
        - y_true: np.array, targets of shape=(events,)
        - fname: Path, the output plot file name
        - mis: np.array, the mutual information for each feature against
               validation set labels, of shape=(nb feats,). If it is not `None`,
               increase transparency of features showing a MI score less than a
               threshold `t` value.
        - t: float, the mi threshold value
        - ffeat: int, first progressive feature to plot
    """
    nrow = 4
    ncol = 4
    print(f"Plotting features {ffeat}-{ffeat+nrow*ncol-1} ...")
    fig = plt.figure(figsize=[6.4 * nrow, 4.8 * ncol])
    axs = fig.subplots(nrow, ncol)
    bins = np.linspace(-3, 3, 101)
    # feature standardization
    mus = feats.mean(-1, keepdims=True)
    sigmas = feats.std(-1, keepdims=True)
    feats = (feats - mus) / sigmas

    up_feats = feats[y_true.astype(bool)]
    down_feats = feats[~y_true.astype(bool)]

    alphas = np.where(mis > t, 1, 0.3) if mis is not None else np.ones_like(mis)

    for r in range(nrow):
        for c in range(ncol):
            n = r * ncol + c + ffeat
            axs[r, c].hist(
                up_feats[:, n],
                bins=bins,
                color="green",
                histtype="step",
                label="merge",
                alpha=alphas[n],
            )
            axs[r, c].hist(
                down_feats[:, n],
                bins=bins,
                color="red",
                histtype="step",
                label="not merge",
                alpha=alphas[n],
            )
            axs[r, c].title.set_text(f"feature {n}, MI= {mis[n]:.3f}")
    axs[0, 0].legend()
    plt.savefig(fname, bbox_inches="tight")
    plt.close()


def main(setup):
    """
    Main function: loads the validation set, the network and plots the
    histograms.

    Parameters
    ----------
        - setup: dict, the loaded settings
    """
    # load dataset
    dataset_dir = Path("../dataset/training")
    val_generator = build_dataset_from_np(setup, dataset_dir)[1]

    # load network
    network = load_network(setup, setup["output"] / setup["test"]["checkpoint"])
    stack = tf.keras.Sequential([l for l in network.layers if "mha_0" in l.name])
    # stack.add(ReduceMax(axis=1, name="reduce_max"))
    stack.build((1, None, setup["model"]["f_dims"]))
    stack.summary()

    # forward pass
    fname = setup["output"] / "cluster_features.npy"
    if fname.is_file():
        y_pred = np.load(fname, allow_pickle=True)
    else:
        y_pred = [stack.predict(inp[None])[0] for inp in tqdm(val_generator.inputs)]
        y_pred = np.array(y_pred, dtype=object)
        np.save(fname, y_pred)
    feats = np.stack([ev.max(0) for ev in y_pred], axis=0)
    y_true = val_generator.targets
    assert (
        feats.shape[0] == y_true.shape[0]
    ), f"found shapes {feats.shape[0]} and {y_true.shape[0]}"

    # mutual information
    mis = mutual_info_classif(feats, y_true).flatten()
    best_feat = np.argmax(mis)
    print(f"Maximum MI {mis.max():.3f} for feature {best_feat}")
    plot_clusters(
        val_generator.inputs,
        y_pred,
        y_true,
        best_feat,
        setup["output"] / f"plots/cluster_best_feature.png",
    )

    # histogramming
    print(f"Histogramming on {y_true.shape[0]} points")
    for i in range(0, y_pred[0].shape[1], 16):
        plot_features(
            feats,
            y_true,
            setup["output"] / f"plots/cluster_features{i}-{i+15}.png",
            mis=mis,
            ffeat=i,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, help="the model folder")
    args = parser.parse_args()

    # load the runcard
    setup = load_runcard(args.model / "runcard.yaml")
    modify_runcard(setup)
    preconfig_tf(setup)

    start = tm()
    main(setup)
    print(f"Program done in {tm()-start}s")
