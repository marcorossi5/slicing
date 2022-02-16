import os
import argparse
from pathlib import Path
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

def config_init():
    parser = argparse.ArgumentParser(
        """
    Example script to train RandLA-Net for slicing
    """
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        dest="data",
        help="The test set path model.",
    )
    parser.add_argument(
        "--save_dataset",
        help="Save training dataset in folder specified by runcard",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--save_train_np",
        help="Save balanced training dataset",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--save_dataset_test",
        help="Save training dataset in folder specified by runcard",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--load_dataset",
        help="Load training dataset from folder specified by runcard",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--load_dataset_test",
        help="Load test dataset from folder specified by runcard",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--debug",
        help="Run TensorFlow eagerly",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--force",
        "-f",
        help="Overwrite existing files if present",
        action="store_true",
        dest="force",
    )
    parser.add_argument(
        "--model", "-m", type=str, default=None, help="The input model."
    )
    parser.add_argument("--output", "-o", help="Output folder", type=Path, default=None)
    parser.add_argument(
        "runcard",
        action="store",
        nargs="?",
        type=Path,
        default=None,
        help="A yaml file with the setup.",
    )
    parser.add_argument(
        "--no_graphics",
        help="DO not display figures via GUI",
        action="store_true",
        default=False,
    )
    parser.add_argument("--just_train", action="store_true", help="Just train with ../dataset/train folder")
    args = parser.parse_args()
    # check that input is coherent
    if (not args.model and not args.runcard) or (args.model and args.runcard):
        raise ValueError("Invalid options: requires either input runcard or model.")
    elif args.runcard and not args.runcard.is_file():
        raise ValueError("Invalid runcard: not a file.")
    if args.force:
        print("WARNING: Running with --force option will overwrite existing model")
    if (args.save_dataset and args.load_dataset) or (
        args.save_dataset_test and args.load_dataset_test
    ):
        raise ValueError("Invalid options: requires either save or load dataset.")
    return args
