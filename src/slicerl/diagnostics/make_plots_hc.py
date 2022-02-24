import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from seaborn import heatmap
from slicerl import PACKAGE
from slicerl.utils.diagnostics import cmap, norm

logger = logging.getLogger(PACKAGE + ".diagnostics")

def make_plots(generator, folder):
    """
    Additional plots for HC-Net diagnostics.

    Parameters
    ----------
        - generator: HCEventDataset, the test generator
        - folder: Path, the output folder
    """
    # add functions here
    plot_confusion_matrix(generator, folder)
    plot_views(generator, folder)
    plot_network_activations(generator, folder)


# ======================================================================
def plot_confusion_matrix(generator, folder):
    y_true = np.concatenate([p.status for ev in generator.events for p in ev.planes])
    y_pred = np.concatenate([p.ordered_mc_idx for ev in generator.events for p in ev.planes])
    nb_classes = generator.y_pred.all_y_pred[0].shape[-1]

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(nb_classes), normalize='all')
    plt.figure(figsize=(9, 9))
    ax = heatmap(cm)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    fname = folder / f"confusion_matrix.png"
    logger.info(f"Saving plot at {fname} ")
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()


# ======================================================================
def plot_views(generator, folder):
    ev = generator.events[0]
    ps = ["U", "V", "W"]
    for p, plane in zip(ps, ev.planes):
        plt.rcParams.update({"font.size": 20})
        fig = plt.figure(figsize=([6.4*2.5, 4.8*1.25]))
        ax = fig.add_subplot(121)
        ax.scatter(plane.point_cloud[1]*1000, plane.point_cloud[2]*1000, s=3, c=plane.status, cmap=cmap, norm=norm)
        ax.title.set_text(f"Plane {p}: network predictions")
        ax.set_xlabel("x [cm]")
        ax.set_ylabel("z [cm]")

        ax = fig.add_subplot(122)
        ax.scatter(plane.point_cloud[1]*1000, plane.point_cloud[2]*1000, s=3, c=plane.ordered_mc_idx, cmap=cmap, norm=norm)
        ax.title.set_text(f"Plane {p}: MC Truths")
        ax.set_xlabel("x [cm]")
        ax.set_ylabel("z [cm]")

        fname = folder / f"pview_{p}.png"
        logger.info(f"Saving plot at {fname} ")
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        plt.close()


# ======================================================================
def plot_network_activations(generator, folder):
    y_sparses = generator.y_pred.all_y_pred[:3]
    nb_classes = y_sparses[0].shape[-1]
    bins = np.linspace(-0.5, nb_classes - 0.5, nb_classes + 1)
    ps = ["U", "V", "W"]
    for p, y_sparse in zip(ps, y_sparses):
        fig = plt.figure()
        ax = fig.add_subplot(2,1,1)
        ax.title.set_text("Network activation")
        ax.plot(y_sparse[:30,].T, lw=0.5)
        ax = fig.add_subplot(2,1,2)
        idx = np.argmax(y_sparse,axis=1)
        h, _ = np.histogram(idx, bins=bins)
        ax.hist(bins[:-1], bins, weights=h)
        fname = folder / f"pview_{p}_activations.png"
        logger.info(f"Saving plot at {fname} ")
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        plt.close()