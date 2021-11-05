# This file is part of SliceRL by M. Rossi
from slicerl.FFNN import FFNN
from slicerl.CMANet import CMANet
from slicerl.losses import dice_loss
from slicerl.build_dataset import dummy_dataset, load_events, get_generator
from slicerl.diagnostics import (
    plot_plane_view,
    plot_slice_size,
    plot_multiplicity,
    plot_test_beam_metrics,
    plot_histogram,
    plot_graph,
)
from slicerl.callbacks import ExtendedTensorBoard
import os
from copy import deepcopy
from time import time as tm
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as tfK
from tensorflow.keras.callbacks import (
    TensorBoard,
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping,
)
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad

from hyperopt import STATUS_OK


def load_network(setup, checkpoint_filepath=None):
    """
    Load network from config dic, compile and, if necessary, load weights.

    Parameters
    ----------
        - setup               : dict, config dict
        - checkpoint_filepath : str, model checkpoint path

    Returns
    -------
        - AbstractNet derived class
    """
    net_dict = {
        "batch_size": setup["model"]["batch_size"],
        "use_bias": setup["model"]["use_bias"],
        "f_dims": setup["model"]["f_dims"],
    }
    if setup["model"]["net_type"] == "FF":
        net = FFNN(name="FFNN", **net_dict)
    elif setup["model"]["net_type"] == "CMA":
        net = CMANet(name="CMANet", **net_dict)
    else:
        raise ValueError(f"Unrecognised network: got {setup['model']['net_type']}")

    lr = setup["train"]["lr"]
    if setup["train"]["optimizer"] == "Adam":
        opt = Adam(learning_rate=lr)
    elif setup["train"]["optimizer"] == "SGD":
        opt = SGD(learning_rate=lr)
    elif setup["train"]["optimizer"] == "RMSprop":
        opt = RMSprop(learning_rate=lr)
    elif setup["train"]["optimizer"] == "Adagrad":
        opt = Adagrad(learning_rate=lr)

    if setup["train"]["loss"] == "xent":
        loss = tf.keras.losses.BinaryCrossentropy(name="xent")
    elif setup["train"]["loss"] == "hinge":
        loss = tf.keras.losses.Hinge(name="hinge")
    elif setup["train"]["loss"] == "dice":
        loss = dice_loss
    else:
        raise NotImplementedError("Loss function not implemented")

    net.compile(
        loss=loss,
        optimizer=opt,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            tf.keras.metrics.Precision(name="prec"),
            tf.keras.metrics.Recall(name="rec"),
        ],
        run_eagerly=setup.get("debug"),
    )

    # if not setup["scan"]:
    #     net.model().summary()

    if checkpoint_filepath:
        # dummy forward pass to build the layers
        dummy_generator = dummy_dataset(setup["model"]["net_type"], setup["model"]["f_dims"])
        net.evaluate(dummy_generator[0][0], verbose=0)

        print(f"[+] Loading weights at {checkpoint_filepath}")
        net.load_weights(checkpoint_filepath)
    return net


# ======================================================================
def build_and_train_model(setup, generators):
    """
    Train a model. If setup['scan'] is True, then perform hyperparameter search.

    Parameters
    ----------
        - setup      : dict
        - generators : list, of EventDataset with train and val generators

    Retruns
    -------
        network model if scan is False, else dict with loss and status keys.
    """
    tfK.clear_session()
    if setup["scan"]:
        batch_size = setup["model"]["batch_size"]
        loss = setup["train"]["loss"]
        lr = setup["train"]["lr"]
        opt = setup["train"]["optimizer"]
        print(f"{{batch_size: {batch_size}, loss: {loss}, lr: {lr}, opt: {opt}}}")

    train_generator, val_generator = generators

    initial_weights = setup["train"]["initial_weights"]
    if initial_weights and os.path.isfile(initial_weights):
        print(f"[+] Found Initial weights configuration at {initial_weights} ... ")
    net = load_network(setup, initial_weights)

    logdir = setup["output"].joinpath(f"logs/{tm()}").as_posix()
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
            write_graph=False,
            write_images=True,
            histogram_freq=setup["train"]["hist_freq"],
            profile_batch=5,
        )

    callbacks.append(tboard)

    """
    truths = train_generator.targets.astype(bool)
    out = np.stack([inp[[0,-1],3:5] for inp in train_generator.inputs])
    sdots = np.abs(out.prod(1).sum(-1))
    import matplotlib.pyplot as plt
    bins = np.linspace(0, 1, 101)
    trues = sdots[truths]
    falses = sdots[~truths]
    t = 0.75
    tp = np.count_nonzero(trues > t)
    tn = np.count_nonzero(falses < t)
    fn = np.count_nonzero(trues < t)
    fp = np.count_nonzero(falses > t)
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    print(f"acc: {acc}, prec: {prec}, rec: {rec}")
    
    strue, _ = np.histogram(trues, bins)
    sfalse, _ = np.histogram(falses, bins)
    plt.hist(bins[:-1], bins, weights=strue, color="green", histtype='step')
    plt.hist(bins[:-1], bins, weights=sfalse, color="red", histtype='step')
    plt.yscale('log')
    plt.show()
    """

    print(f"[+] Train for {setup['train']['epochs']} epochs ...")
    r = net.fit(
        train_generator,
        epochs=setup["train"]["epochs"],
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=2,
    )

    if setup["scan"]:
        net.load_weights(checkpoint_filepath)
        loss, acc, prec, rec = net.evaluate(val_generator, verbose=0)
        print(
            f"Evaluate model instance: [loss: {loss:.5f}, acc: {acc:.5f}, prec: {prec:.5f}, rec: {rec:.5f}]"
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


# ----------------------------------------------------------------------
def inference(setup, show_graph=False, no_graphics=False):
    tfK.clear_session()
    print("[+] done with training, load best weights")
    checkpoint_filepath = setup["output"].joinpath("network.h5")
    net = load_network(setup, checkpoint_filepath.as_posix())

    # TODO: this must be a generator with is_training and without the splitting
    # but it's not mandatory
    # results = net.evaluate(test_generator)
    # print(
    #     f"Test loss: {results[0]:.5f} \t test accuracy: {results[1]} \t test precision: {results[2]} \t test recall: {results[3]}"
    # )

    min_hits = setup["test"]["min_hits"]
    cthreshold = setup["test"]["cthreshold"]
    events = load_events(setup["test"]["fn"], setup["test"]["nev"], min_hits)
    test_generator = get_generator(
        events,
        setup["model"]["net_type"],
        setup["model"]["test_batch_size"],
        min_hits=min_hits,
        cthreshold=cthreshold,
    )

    # collect statistics
    hist_true = []
    hist_pred = []
    test_batch_size = 1 if setup["model"]["net_type"] == "CMA" else setup["model"]["test_batch_size"]
    y_pred = net.get_prediction(
        test_generator,
        test_batch_size,
        threshold=setup["model"]["threshold"],
    )
    test_generator.events = y_pred

    hist_true.append([trg.flatten() for trg in test_generator.targets])
    hist_pred.append([pred.flatten() for pred in y_pred.all_y_pred])

    for i, ev in enumerate(test_generator.events):
        do_visual_checks(ev, i, setup["output"], no_graphics)
        if i > 10:
            break
    """
    plot_slice_size(test_generator.events, setup["output"].joinpath("plots"))
    plot_multiplicity(test_generator.events, setup["output"].joinpath("plots"))
    plot_test_beam_metrics(test_generator.events, setup["output"].joinpath("plots"))

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
            setup["output"].joinpath("plots"),
        )
    """

    # plot histogram of the network decisions
    # hist_true = [trg.flatten() for trg in test_generator.targets]
    # hist_pred = [pred.flatten() for pred in y_pred.all_y_pred]
    for htrue, hpred in zip(hist_true, hist_pred):
        plot_histogram(htrue, hpred, setup["output"].joinpath("plots"))

    exit("build_model.py l.297")

    if show_graph:
        plot_graph(
            test_generator.get_pc(0),
            deepcopy(y_pred.get_graph(0)),
            y_pred.get_status(0),
            setup["output"].joinpath("plots"),
        )


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

    plt.figure(figsize=(6.4 * 3, 4.8 * 5))
    plt.subplot(531)
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
    plt.tick_params(axis="both", labelleft=True, labelbottom=False)
    plt.subplot(532)
    plt.title("V plane")
    plt.scatter(
        ev.V.calohits[1] * 1000,
        ev.V.calohits[2] * 1000,
        s=0.5,
        c=statusV % 128,
        norm=norm,
        cmap=cmap,
    )
    plt.tick_params(axis="both", labelbottom=False)
    plt.subplot(533)
    plt.title("W plane")
    plt.scatter(
        ev.W.calohits[1] * 1000,
        ev.W.calohits[2] * 1000,
        s=0.5,
        c=statusW % 128,
        norm=norm,
        cmap=cmap,
    )
    plt.tick_params(axis="both", labelbottom=False)
    plt.subplot(534)
    plt.ylabel("Pandora")
    plt.scatter(
        ev.U.calohits[1] * 1000,
        ev.U.calohits[2] * 1000,
        s=0.5,
        c=ev.U.ordered_pndr_idx,
        norm=norm,
        cmap=cmap,
    )
    plt.tick_params(axis="both", labelbottom=False)
    plt.subplot(535)
    plt.scatter(
        ev.V.calohits[1] * 1000,
        ev.V.calohits[2] * 1000,
        s=0.5,
        c=ev.V.ordered_pndr_idx,
        norm=norm,
        cmap=cmap,
    )
    plt.tick_params(axis="both", labelbottom=False)
    plt.subplot(536)
    plt.scatter(
        ev.W.calohits[1] * 1000,
        ev.W.calohits[2] * 1000,
        s=0.5,
        c=ev.W.ordered_pndr_idx,
        norm=norm,
        cmap=cmap,
    )
    plt.tick_params(axis="both", labelbottom=False)
    plt.subplot(537)
    plt.ylabel("pfos")
    plt.scatter(
        ev.U.calohits[1] * 1000,
        ev.U.calohits[2] * 1000,
        s=0.5,
        c=pfoU % 128,
        norm=norm,
        cmap=cmap,
    )
    plt.tick_params(axis="both", labelbottom=False)
    plt.subplot(538)
    plt.scatter(
        ev.V.calohits[1] * 1000,
        ev.V.calohits[2] * 1000,
        s=0.5,
        c=pfoV % 128,
        norm=norm,
        cmap=cmap,
    )
    plt.tick_params(axis="both", labelbottom=False)
    plt.subplot(539)
    plt.scatter(
        ev.W.calohits[1] * 1000,
        ev.W.calohits[2] * 1000,
        s=0.5,
        c=pfoW % 128,
        norm=norm,
        cmap=cmap,
    )
    plt.tick_params(axis="both", labelbottom=False)
    plt.subplot(5, 3, 10)
    plt.ylabel("Test beam")
    plt.scatter(
        ev.U.calohits[1] * 1000,
        ev.U.calohits[2] * 1000,
        s=0.5,
        c=ev.U.calohits[9],
        norm=norm,
        cmap=cmap,
    )
    plt.tick_params(axis="both", labelbottom=False)
    plt.subplot(5, 3, 11)
    plt.scatter(
        ev.V.calohits[1] * 1000,
        ev.V.calohits[2] * 1000,
        s=0.5,
        c=ev.V.calohits[9],
        norm=norm,
        cmap=cmap,
    )
    plt.tick_params(axis="both", labelbottom=False)
    plt.subplot(5, 3, 12)
    plt.scatter(
        ev.W.calohits[1] * 1000,
        ev.W.calohits[2] * 1000,
        s=0.5,
        c=ev.W.calohits[9],
        norm=norm,
        cmap=cmap,
    )
    plt.tick_params(axis="both", labelbottom=False)
    plt.subplot(5, 3, 13)
    plt.ylabel("Input clusters")
    plt.scatter(
        ev.U.calohits[1] * 1000,
        ev.U.calohits[2] * 1000,
        s=0.5,
        c=ev.U.calohits[5] % 128,
        norm=norm,
        cmap=cmap,
    )
    plt.tick_params(axis="both", labelbottom=True)
    plt.subplot(5, 3, 14)
    plt.scatter(
        ev.V.calohits[1] * 1000,
        ev.V.calohits[2] * 1000,
        s=0.5,
        c=ev.V.calohits[5] % 128,
        norm=norm,
        cmap=cmap,
    )
    plt.tick_params(axis="both", labelbottom=True)
    plt.subplot(5, 3, 15)
    plt.scatter(
        ev.W.calohits[1] * 1000,
        ev.W.calohits[2] * 1000,
        s=0.5,
        c=ev.W.calohits[5] % 128,
        norm=norm,
        cmap=cmap,
    )
    plt.tick_params(axis="both", labelbottom=True)

    if no_graphics:
        fname = output_dir.joinpath(f"plots/visual_check_{evno}.png")
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()
    plt.show()
