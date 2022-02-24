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
from slicerl.diagnostics.make_plots import make_plots

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

    make_plots(setup, test_generator)
