import os
import random
import tensorflow as tf
import numpy as np


# ======================================================================
def config_tf(setup):
    """ Set the host device for tensorflow. """
    os.environ["CUDA_VISIBLE_DEVICES"] = setup.get("gpu")
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


# ======================================================================
def set_manual_seed(seed):
    """ Set libraries random seed for reproducibility. """
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


EPS = np.finfo(np.float32).eps

TF_DTYPE_INT = tf.int32
TF_DTYPE = tf.float32

NP_DTYPE_INT = np.int32
NP_DTYPE = np.float32


def float_me(x):
    return tf.cast(x, dtype=TF_DTYPE)


def int_me(x):
    return tf.cast(x, dtype=TF_DTYPE_INT)

EPS_TF = float_me(EPS)
