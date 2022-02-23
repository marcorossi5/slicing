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
def get_activation(act):
    """ Get activation from string. """
    try:
        fn = tf.keras.activations.get(act)
        activation = lambda x: fn(x)
    except:
        if act == "lrelu":
            activation = lambda x: tf.keras.activations.relu(x, alpha=0.0)
        else:
            raise ValueError(f"activation not recognized by keras, found {act}")
    return activation


# ======================================================================
def load_network_cm(msetup, loss_str):
    """
    Load CM-Net from config dict.

    Parameters
    ----------
        - msetup: dict, model config dict
        - loss_str: str, the loss function string representation

    Returns
    -------
        - Network
        - loss
    """
    
    net_dict = {
        "f_dims": msetup["f_dims"],
        "nb_mha_heads": msetup["nb_mha_heads"],
        "mha_filters": msetup["mha_filters"],
        "nb_fc_heads": msetup["nb_fc_heads"],
        "fc_filters": msetup["fc_filters"],
        "batch_size": msetup["batch_size"],
        "activation": get_activation(msetup["activation"]),
        "use_bias": msetup["use_bias"],
    }
    network = CMNet(name="CM-Net", **net_dict)

    if loss_str == "xent":
        loss = tf.keras.losses.BinaryCrossentropy(name="xent")
    elif loss_str == "hinge":
        loss = tf.keras.losses.Hinge(name="hinge")
    elif loss_str == "dice":
        loss = dice_loss
    else:
        raise NotImplementedError("Loss function not implemented")

    return network, loss


# ======================================================================
def load_network_hc(msetup):
    """
    Load HC-Net from config dict.

    Parameters
    ----------
        - msetup: dict, model config dict

    Returns
    -------
        - Network
        - loss
    """
    net_dict = {
        "units": msetup["units"],
        "f_dims": msetup["f_dims"],
        "nb_mha_heads": msetup["nb_mha_heads"],
        "mha_filters": msetup["mha_filters"],
        "kernel_transformation": msetup["kernel_transformation"],
        "projection_matrix_type": msetup["projection_matrix_type"],
        "nb_fc_heads": msetup["nb_fc_heads"],
        "fc_filters": msetup["fc_filters"],
        "batch_size": msetup["batch_size"],
        "activation": get_activation(msetup["activation"]),
        "use_bias": msetup["use_bias"],
    }
    network = HCNet(name="HC-Net", **net_dict)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
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
        network, loss = load_network_cm(setup["model"], setup["train"]["loss"])
    elif modeltype == "HC":
        network, loss = load_network_hc(setup["model"])
    else:
        raise NotImplementedError(f"Modeltype not implemented, got {modeltype}")

    metrics = get_metrics(modeltype, setup)

    network.compile(
        loss=loss,
        optimizer=opt,
        metrics=metrics,
        run_eagerly=setup.get("debug"),
    )

    if checkpoint_filepath:
        logger.info(f"Loading weights at {checkpoint_filepath}")
        network.load_weights(checkpoint_filepath)
    exit()
    return network
