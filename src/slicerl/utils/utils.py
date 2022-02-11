# This file is part of SliceRL by M. Rossi
""" This module contains utility functions of general interest. """
import yaml
from hyperopt import hp
import tensorflow as tf
import numpy as np
import random

# ======================================================================
def load_runcard(runcard_file):
    """Load runcard from yaml file."""
    with open(runcard_file, "r") as stream:
        runcard = yaml.load(stream, Loader=yaml.FullLoader)
    runcard["scan"] = False
    for key, value in runcard.items():
        if isinstance(value, dict):
            for k, v in value.items():
                if "hp." in str(v):
                    runcard[key][k] = eval(v)
                    runcard["scan"] = True
        else:
            if "hp." in str(value):
                runcard[key][k] = eval(value)
                runcard["scan"] = True
    return runcard

# ======================================================================
def set_manual_seed(seed):
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

