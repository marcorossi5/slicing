""" This module contains setups to configure tensorflow. """
import os
import random
import numpy as np
import tensorflow as tf

TF_DTYPE_INT = tf.int32
TF_DTYPE = tf.float32

NP_DTYPE_INT = np.int32
NP_DTYPE = np.float32


def float_me(x):
    return tf.cast(x, dtype=TF_DTYPE)


def int_me(x):
    return tf.cast(x, dtype=TF_DTYPE_INT)


EPS = np.finfo(np.float32).eps
EPS_TF = float_me(EPS)


def config_tf(setup):
    """Set the host device for tensorflow."""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if setup["debug"]:
        print("[+] Run all tf functions eagerly")
        tf.config.run_functions_eagerly(True)
        # tf.data.experimental.enable_debug_mode()


def set_manual_seed(seed):
    """Set libraries random seed for reproducibility."""
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
