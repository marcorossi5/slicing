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
from slicerl.config import config_tf
from slicerl.utils.utils import load_runcard, modify_runcard
from slicerl.build_dataset import build_dataset_from_np
from slicerl.build_model import load_network
from slicerl.CMNet import ReduceMax


def plot_features(feats, y_true, fname, ffeat=0):
    """
    Plots the extracted features.

    Parameters
    ----------
        - feats: np.array, features array of shape=(events, nb feats)
        - y_true: np.array, targets of shape=(events,)
        - fname: Path, the output plot file name
        - ffeat: int, first progressive feature to plot
    """
    nrow = 4
    ncol = 4
    print(f"Plotting features {ffeat}-{ffeat+nrow*ncol-1}...")
    fig = plt.figure(figsize=[6.4 * nrow, 4.8 * ncol])
    axs = fig.subplots(nrow, ncol)
    bins = np.linspace(-3, 3, 101)
    # feature standardization
    mus = feats.mean(-1, keepdims=True)
    sigmas = feats.std(-1, keepdims=True)
    feats = (feats - mus) / sigmas

    up_feats = feats[y_true.astype(bool)]
    down_feats = feats[~y_true.astype(bool)]

    for r in range(nrow):
        for c in range(ncol):
            n = r * ncol + c + ffeat
            axs[r, c].hist(
                up_feats[:, n], bins=bins, color="green", histtype="step", label="merge"
            )
            axs[r, c].hist(
                down_feats[:, n],
                bins=bins,
                color="red",
                histtype="step",
                label="not merge",
            )
            axs[r, c].title.set_text(f"feature {n}")
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
    val_generator = build_dataset_from_np(setup, Path("../dataset/training"))[1]

    # load network
    network = load_network(setup, setup["output"] / setup["test"]["checkpoint"])
    stack = tf.keras.Sequential([l for l in network.layers if "mha" in l.name])
    stack.add(ReduceMax(axis=1, name="reduce_max"))
    stack.build((1, None, setup["model"]["f_dims"]))
    stack.summary()

    # forward pass
    fname = setup["output"] / "cluster_features.npy"
    if fname.is_file():
        feats = np.load(fname)
    else:
        feats = [stack.predict(inp[None]) for inp in tqdm(val_generator.inputs)]
        feats = np.concatenate(feats)
        np.save(fname, feats)

    # histogramming
    print(f"Histogramming on {val_generator.targets.shape[0]} points")
    y_true = val_generator.targets
    for i in range(0, 128, 16):
        plot_features(
            feats,
            y_true,
            setup["output"] / f"plots/cluster_features{i}-{i+15}.png",
            ffeat=i,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, help="the model folder")
    args = parser.parse_args()

    # load the runcard
    setup = load_runcard(args.model / "runcard.yaml")
    modify_runcard(setup)
    config_tf(setup)

    start = tm()
    main(setup)
    print(f"Program done in {tm()-start}s")
