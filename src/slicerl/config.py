import tensorflow as tf
import numpy as np

EPS = np.finfo(np.float64).eps

TF_DTYPE_INT = tf.int32
TF_DTYPE     = tf.float32

NP_DTYPE_INT = np.int32
NP_DTYPE     = np.float32

def float_me(x):
    return tf.constant(x, dtype=TF_DTYPE)

def int_me(x):
    return tf.constant(x, dtype=TF_DTYPE_INT)