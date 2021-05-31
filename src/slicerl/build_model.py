# This file is part of SliceRL by M. Rossi
from slicerl.SEACNet import SeacNet
from slicerl.build_dataset import dummy_dataset
from slicerl.diagnostics import (
    plot_plane_view,
    plot_slice_size,
    plot_multiplicity,
    plot_histogram,
    plot_graph,
)

import os
from copy import deepcopy
from time import time as tm

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
        - SeacNet
    """

    net_dict = {
        "K": setup["model"]["K"],
        "locse_nb_layers": setup["model"]["locse_nb_layers"],
        "use_bias": setup["model"]["use_bias"],
        "use_bnorm": setup["model"]["use_bnorm"],
    }
    net = SeacNet(name="SEAC-Net", **net_dict)

    lr = setup["train"]["lr"]
    if setup["train"]["optimizer"] == "Adam":
        opt = Adam(lr=lr)
    elif setup["train"]["optimizer"] == "SGD":
        opt = SGD(lr=lr)
    elif setup["train"]["optimizer"] == "RMSprop":
        opt = RMSprop(lr=lr)
    elif setup["train"]["optimizer"] == "Adagrad":
        opt = Adagrad(lr=lr)

    if setup["train"]["loss"] == "xent":
        loss = tf.keras.losses.BinaryCrossentropy(name="xent")
    elif setup["train"]["loss"] == "hinge":
        loss = tf.keras.losses.Hinge(name="hinge")
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
    if not setup["scan"]:
        net.model().summary()
    # tf.keras.utils.plot_model(net.model(), to_file=f"{setup['output']}/Network.png", expand_nested=True, show_shapes=True)

    if checkpoint_filepath:
        print(f"[+] Loading weights at {checkpoint_filepath}")
        dummy_generator = dummy_dataset(
            setup["model"]["nb_classes"], setup["model"]["K"]
        )
        net.evaluate(dummy_generator, verbose=0)
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
        use_bnorm = setup["model"]["use_bnorm"]
        K = setup["model"]["K"]
        use_bias = setup["model"]["use_bias"]
        lr = setup["train"]["lr"]
        opt = setup["train"]["optimizer"]
        print(
            f"{{use_bnorm: {use_bnorm}, K: {K}, use_bias: {use_bias}, lr: {lr}, opt: {opt}}}"
        )

    train_generator, val_generator = generators

    initial_weights = setup["train"]["initial_weights"]
    if initial_weights and os.path.isfile(initial_weights):
        print(
            f"[+] Found Initial weights configuration at {initial_weights} ... "
        )
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
            factor=0.5,
            mode="max",
            verbose=1,
            patience=setup["train"]["patience"],
            min_lr=setup["train"]["min_lr"],
        ),
        # EarlyStopping(
        #     monitor='val_acc',
        #     min_delta=0.001,
        #     mode='max',
        #     patience=15,
        #     restore_best_weights=True
        # )
    ]
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


def inference(setup, test_generator, show_graph=False):
    tfK.clear_session()
    print("[+] done with training, load best weights")
    checkpoint_filepath = setup["output"].joinpath("network.h5")
    net = load_network(setup, checkpoint_filepath.as_posix())

    results = net.evaluate(test_generator)
    print(
        f"Test loss: {results[0]:.5f} \t test accuracy: {results[1]} \t test precision: {results[2]} \t test recall: {results[3]}"
    )

    y_pred = net.get_prediction(
        test_generator.prep_inputs,
        test_generator.knn_idxs,
        threshold=setup["model"]["threshold"],
    )

    test_generator.events = y_pred

    plot_slice_size(test_generator.events, setup["output"].joinpath("plots"))
    plot_multiplicity(test_generator.events, setup["output"].joinpath("plots"))

    n = min(10, len(test_generator))
    for i in range(n):
        pc = test_generator.get_pc(i)  # shape=(N,2)
        pc_pred = y_pred.get_status(i)  # shape=(N,)
        pc_test = test_generator.get_targets(i)  # shape=(N,)
        plot_plane_view(
            pc, pc_pred, pc_test, i, setup["output"].joinpath("plots")
        )

    # plot histogram of the network decisions
    hist_true = [trg.flatten() for trg in test_generator.prep_targets]
    hist_pred = [pred.flatten() for pred in y_pred.preds]
    plot_histogram(hist_true, hist_pred, setup["output"].joinpath("plots"))

    if show_graph:
        plot_graph(
            test_generator.get_pc(0),
            deepcopy(y_pred.get_graph(0)),
            y_pred.get_status(0),
            setup["output"].joinpath("plots"),
        )
