import logging
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad
from slicerl import PACKAGE
from slicerl.networks.CMNet import CMNet
from slicerl.networks.HCNet import HCNet
from slicerl.utils.losses import dice_loss

logger = logging.getLogger(PACKAGE)


def get_metrics_cm():
    return [
        tf.keras.metrics.BinaryAccuracy(name="acc"),
        tf.keras.metrics.Precision(name="prec"),
        tf.keras.metrics.Recall(name="rec"),
    ]


# ======================================================================
def get_metrics_hc(nb_classes):
    return [
        tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5"),
        # tf.keras.metrics.MeanIoU(nb_classes, name="MIoU")
    ]


# ======================================================================
def get_metrics(modeltype, setup):
    if modeltype == "CM":
        return get_metrics_cm()
    elif modeltype == "HC":
        return get_metrics_hc(setup["model"]["units"])


# ======================================================================
def load_network_cm(setup):
    """
    Load network from config dic, compile and, if necessary, load weights.

    Parameters
    ----------
        - setup: dict, config dict
        - checkpoint_filepath: Path, model checkpoint path

    Returns
    -------
        - Network
        - loss
    """
    net_dict = {
        "batch_size": setup["model"]["batch_size"],
        "f_dims": setup["model"]["f_dims"],
        "use_bias": setup["model"]["use_bias"],
        "activation": lambda x: tf.keras.activations.relu(x, alpha=0.),
    }
    network = CMNet(name="CM-Net", **net_dict)

    if setup["train"]["loss"] == "xent":
        loss = tf.keras.losses.BinaryCrossentropy(name="xent")
    elif setup["train"]["loss"] == "hinge":
        loss = tf.keras.losses.Hinge(name="hinge")
    elif setup["train"]["loss"] == "dice":
        loss = dice_loss
    else:
        raise NotImplementedError("Loss function not implemented")

    return network, loss


# ======================================================================
def load_network_hc(setup):
    net_dict = {
        "units": setup["model"]["units"],
        "batch_size": setup["model"]["batch_size"],
        "f_dims": setup["model"]["f_dims"],
        "use_bias": setup["model"]["use_bias"],
        "activation": lambda x: tf.keras.activations.relu(x, alpha=0.2),
    }
    network = HCNet(name="HC-Net", **net_dict)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(name="catxent")
    return network, loss


# ======================================================================
def load_and_compile_network(setup, checkpoint_filepath=None):
    modeltype = setup["model"]["net_type"]

    # optimizer
    lr = setup["train"]["lr"]
    if setup["train"]["optimizer"] == "Adam":
        opt = Adam(learning_rate=lr)
    elif setup["train"]["optimizer"] == "SGD":
        opt = SGD(learning_rate=lr)
    elif setup["train"]["optimizer"] == "RMSprop":
        opt = RMSprop(learning_rate=lr)
    elif setup["train"]["optimizer"] == "Adagrad":
        opt = Adagrad(learning_rate=lr)

    if modeltype == "CM":
        network, loss = load_network_cm(setup)
    elif modeltype == "HC":
        network, loss = load_network_hc(setup)
    else:
        raise NotImplementedError(f"Modeltype not implemented, got {modeltype}")

    metrics = get_metrics(modeltype, setup)

    network.compile(
        loss=loss,
        optimizer=opt,
        metrics=metrics,
        run_eagerly=setup.get("debug"),
    )

    # if checkpoint_filepath:
    #     logger.info(f"Loading weights at {checkpoint_filepath}")
    #     network.load_weights(checkpoint_filepath)
    return network
