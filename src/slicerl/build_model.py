# This file is part of SliceRL by M. Rossi
import logging
from pathlib import Path
from copy import deepcopy
from time import time as tm
import tensorflow.keras.backend as tfK
from tensorflow.keras.callbacks import (
    TensorBoard,
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping,
)

from hyperopt import STATUS_OK

from slicerl import PACKAGE
from slicerl.utils.configflow import set_manual_seed
from slicerl.networks.inference import get_prediction
from slicerl.networks.load_networks import load_and_compile_network
from slicerl.diagnostics import (
    plot_slice_size,
    plot_multiplicity,
    plot_test_beam_metrics,
    plot_histogram,
)

logger = logging.getLogger(PACKAGE)
logger_hopt = logging.getLogger(PACKAGE + ".hopt")


def train_network(setup, net, generators):
    if setup["scan"]:
        batch_size = setup["model"]["batch_size"]
        loss = setup["train"]["loss"]
        lr = setup["train"]["lr"]
        opt = setup["train"]["optimizer"]
        logging.info(
            f"{{batch_size: {batch_size}, loss: {loss}, lr: {lr}, opt: {opt}}}"
        )

    train_generator, val_generator = generators

    logdir = setup["output"] / f"logs/{tm()}"
    checkpoint_filepath = setup["output"].joinpath(f"network.h5").as_posix()
    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            save_best_only=True,
            mode="max",
            monitor="val_acc",
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_acc",
            factor=0.75,
            mode="max",
            verbose=1,
            patience=setup["train"]["patience"],
            min_lr=setup["train"]["min_lr"],
        ),
    ]
    if setup["train"]["es_patience"]:
        callbacks.append(
            EarlyStopping(
                monitor="val_acc",
                min_delta=0.0001,
                mode="max",
                patience=setup["train"]["es_patience"],
                restore_best_weights=True,
            )
        )

    if setup["scan"]:
        tboard = TensorBoard(log_dir=logdir, profile_batch=0)
    else:
        tboard = TensorBoard(
            log_dir=logdir,
            # write_graph=False,
            # write_images=True,
            # histogram_freq=setup["train"]["hist_freq"],
            # profile_batch=5,
        )

    callbacks.append(tboard)
    logger.info(f"Train for {setup['train']['epochs']} epochs ...")
    net.fit(
        train_generator,
        epochs=setup["train"]["epochs"],
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=2,
    )

    if setup["scan"]:
        net.load_weights(checkpoint_filepath)
        loss, acc, prec, rec = net.evaluate(val_generator, verbose=0)
        logger_hopt.info(
            f"Evaluate model instance: [loss: {loss:.5f}, acc: {acc:.5f}, "
            f"prec: {prec:.5f}, rec: {rec:.5f}]"
        )
        res = {
            "loss": -acc,
            "xent": loss,
            "prec": prec,
            "rec": rec,
            "status": STATUS_OK,
        }
    else:
        res = net
    return res


# ======================================================================
def build_and_train_model(setup, generators, seed=12345):
    """
    Train a model. If setup['scan'] is True, then perform hyperparameter search.

    Parameters
    ----------
        - setup: dict
        - generators: list, of EventDataset with train and val generators
        - seed: int, random generator seed for reproducibility

    Retruns
    -------
        network model if scan is False, else dict with loss and status keys.
    """
    tfK.clear_session()
    set_manual_seed(seed)
    iw = setup["train"]["initial_weights"]
    checkpoint_filepath = Path(iw) if iw is not None else iw
    if checkpoint_filepath and not checkpoint_filepath.is_file():
        raise FileNotFoundError(f"{checkpoint_filepath} no such file or directory")
    net = load_and_compile_network(setup, checkpoint_filepath)
    net.summary()
    return train_network(setup, net, generators)


# ======================================================================
def inference(setup, test_generator, no_graphics=False):
    tfK.clear_session()
    logger.info("Done with training, load best weights")
    checkpoint_filepath = setup["output"] / setup["test"]["checkpoint"]
    if checkpoint_filepath and not checkpoint_filepath.is_file():
        raise FileNotFoundError(f"{checkpoint_filepath} no such file or directory")
    net = load_and_compile_network(setup, checkpoint_filepath)

    y_pred = get_prediction(
        net,
        test_generator,
        threshold=setup["model"]["threshold"],
    )

    test_generator.events = y_pred

    import matplotlib.pyplot as plt
    from slicerl.diagnostics import cmap, norm
    for plane in test_generator.events[0].planes:
        plt.subplot(121)
        plt.scatter(plane.calohits[1], plane.calohits[2], c=plane.status, cmap=cmap, norm=norm, s=3)
        plt.subplot(122)
        plt.scatter(plane.calohits[1], plane.calohits[2], c=plane.ordered_mc_idx, cmap=cmap, norm=norm, s=3)
        plt.show()
    exit()
    for i, ev in enumerate(test_generator.events):
        # TODO: make diagnostics subfolder and jsut call a make_plots wrap function
        # clean this module !!!
        do_visual_checks(ev, i, setup["output"], no_graphics)
        if i > 10:
            break
    # plot histogram of the network decisions
    hist_true = [trg.flatten() for trg in test_generator.targets]
    hist_pred = [pred.flatten() for pred in y_pred.all_y_pred]

    plot_histogram(hist_true, hist_pred, setup["output"] / "plots")
    plot_test_beam_metrics(test_generator.events, setup["output"])
    plot_slice_size(test_generator.events, setup["output"] / "plots")
    plot_multiplicity(test_generator.events, setup["output"] / "plots")

    """
    n = min(30, len(test_generator))
    for i in range(n):
        pc = test_generator.get_pc(i)  # shape=(N,2)
        pc_init = test_generator.events[i].ordered_cluster_idx
        pc_pred = test_generator.get_status(i)  # shape=(N,)
        pc_pndr = test_generator.events[i].ordered_pndr_idx
        pc_test = test_generator.get_targets(i)  # shape=(N,)
        plot_plane_view(
            pc,
            pc_init,
            pc_pred,
            pc_pndr,
            pc_test,
            i,
            setup["output"] / "plots",
        )
    """


# ======================================================================
def do_visual_checks(ev, evno, output_dir, no_graphics):
    import matplotlib.pyplot as plt
    import numpy as np
    from slicerl.diagnostics import norm, cmap
    from copy import deepcopy

    pfoU = deepcopy(ev.U.calohits[-1])
    pfoV = deepcopy(ev.V.calohits[-1])
    pfoW = deepcopy(ev.W.calohits[-1])

    def sort_fn(x):
        all_calo = np.concatenate(
            [ev.U.calohits[-1], ev.V.calohits[-1], ev.W.calohits[-1]]
        )
        return np.count_nonzero(all_calo == x)

    sorted_pfosU = sorted(list(set(ev.U.calohits[-1])), key=sort_fn, reverse=True)
    sorted_pfosV = sorted(list(set(ev.V.calohits[-1])), key=sort_fn, reverse=True)
    sorted_pfosW = sorted(list(set(ev.W.calohits[-1])), key=sort_fn, reverse=True)
    for i, idx in enumerate(sorted_pfosU):
        pfoU[ev.U.calohits[-1] == idx] = i
    for i, idx in enumerate(sorted_pfosV):
        pfoV[ev.V.calohits[-1] == idx] = i
    for i, idx in enumerate(sorted_pfosW):
        pfoW[ev.W.calohits[-1] == idx] = i

    # ----------------------------------------------
    statusU = deepcopy(ev.U.status)
    statusV = deepcopy(ev.V.status)
    statusW = deepcopy(ev.W.status)

    def sort_fn(x):
        all_calo = np.concatenate([ev.U.status, ev.V.status, ev.W.status])
        return np.count_nonzero(all_calo == x)

    sorted_statusU = sorted(list(set(ev.U.status)), key=sort_fn, reverse=True)
    sorted_statusV = sorted(list(set(ev.V.status)), key=sort_fn, reverse=True)
    sorted_statusW = sorted(list(set(ev.W.status)), key=sort_fn, reverse=True)
    for i, idx in enumerate(sorted_statusU):
        statusU[ev.U.status == idx] = i
    for i, idx in enumerate(sorted_statusV):
        statusV[ev.V.status == idx] = i
    for i, idx in enumerate(sorted_statusW):
        statusW[ev.W.status == idx] = i

    plt.figure(figsize=(6.4 * 3, 4.8 * 4))
    plt.subplot(431)
    plt.title("U plane")
    plt.ylabel("output")
    plt.scatter(
        ev.U.calohits[1] * 1000,
        ev.U.calohits[2] * 1000,
        s=0.5,
        c=statusU % 128,
        norm=norm,
        cmap=cmap,
    )
    plt.subplot(432)
    plt.title("V plane")
    plt.scatter(
        ev.V.calohits[1] * 1000,
        ev.V.calohits[2] * 1000,
        s=0.5,
        c=statusV % 128,
        norm=norm,
        cmap=cmap,
    )
    plt.subplot(433)
    plt.title("W plane")
    plt.scatter(
        ev.W.calohits[1] * 1000,
        ev.W.calohits[2] * 1000,
        s=0.5,
        c=statusW % 128,
        norm=norm,
        cmap=cmap,
    )

    plt.subplot(434)
    plt.ylabel("pfos")
    plt.scatter(
        ev.U.calohits[1] * 1000,
        ev.U.calohits[2] * 1000,
        s=0.5,
        c=pfoU % 128,
        norm=norm,
        cmap=cmap,
    )
    plt.subplot(435)
    plt.scatter(
        ev.V.calohits[1] * 1000,
        ev.V.calohits[2] * 1000,
        s=0.5,
        c=pfoV % 128,
        norm=norm,
        cmap=cmap,
    )
    plt.subplot(436)
    plt.scatter(
        ev.W.calohits[1] * 1000,
        ev.W.calohits[2] * 1000,
        s=0.5,
        c=pfoW % 128,
        norm=norm,
        cmap=cmap,
    )

    plt.subplot(437)
    plt.ylabel("Test beam")
    plt.scatter(
        ev.U.calohits[1] * 1000,
        ev.U.calohits[2] * 1000,
        s=0.5,
        c=ev.U.calohits[9],
        norm=norm,
        cmap=cmap,
    )
    plt.subplot(438)
    plt.scatter(
        ev.V.calohits[1] * 1000,
        ev.V.calohits[2] * 1000,
        s=0.5,
        c=ev.V.calohits[9],
        norm=norm,
        cmap=cmap,
    )
    plt.subplot(439)
    plt.scatter(
        ev.W.calohits[1] * 1000,
        ev.W.calohits[2] * 1000,
        s=0.5,
        c=ev.W.calohits[9],
        norm=norm,
        cmap=cmap,
    )

    plt.subplot(4, 3, 10)
    plt.ylabel("Input clusters")
    plt.scatter(
        ev.U.calohits[1] * 1000,
        ev.U.calohits[2] * 1000,
        s=0.5,
        c=ev.U.calohits[5] % 128,
        norm=norm,
        cmap=cmap,
    )
    plt.subplot(4, 3, 11)
    plt.scatter(
        ev.V.calohits[1] * 1000,
        ev.V.calohits[2] * 1000,
        s=0.5,
        c=ev.V.calohits[5] % 128,
        norm=norm,
        cmap=cmap,
    )
    plt.subplot(4, 3, 12)
    plt.scatter(
        ev.W.calohits[1] * 1000,
        ev.W.calohits[2] * 1000,
        s=0.5,
        c=ev.W.calohits[5] % 128,
        norm=norm,
        cmap=cmap,
    )

    if no_graphics:
        fname = output_dir / f"plots/visual_check_{evno}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()
    plt.show()
