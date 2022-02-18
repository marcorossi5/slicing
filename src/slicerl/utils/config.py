"""
    This module reads the command line options and initializes the directory
    structure.
"""
import os
import shutil
import logging
import tensorflow as tf
from slicerl import PACKAGE
from .utils import (
    get_cmd_args,
    check_cmd_args,
    initialize_output_folder,
    check_dataset_directory,
    load_runcard,
    save_runcard,
    modify_runcard,
)

logger = logging.getLogger(PACKAGE)


def preconfig_tf(setup):
    """
    Set the host device for tensorflow. The CUDA_VISIBLE_DEVICES variable must
    be set before prior to allocating any tensors or executing any tf ops.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = setup.get("gpu")
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if setup.get("debug"):
        logger.warning("Run all tf functions eagerly")
        tf.config.run_functions_eagerly(True)
        # tf.data.experimental.enable_debug_mode()


# ======================================================================
def config_init():
    args = get_cmd_args()
    check_cmd_args(args)

    if args.debug:
        logger.setLevel("DEBUG")
        logger.handlers[0].setLevel("DEBUG")

    setup = {"debug": args.debug}
    if args.runcard:
        setup.update(load_runcard(args.runcard))
        initialize_output_folder(args.output, args.force, setup.get("scan"))
        setup["output"] = args.output
        shutil.copyfile(args.runcard, args.output / "input-runcard.yaml")
        save_runcard(args.output / "runcard.yaml", setup)
    elif args.model:
        setup = load_runcard(args.model / "runcard.yaml")
    else:
        raise ValueError("Check inputs, you shouldn't be here !")

    modify_runcard(setup)
    preconfig_tf(setup)

    # check dataset directory structure
    check_dataset_directory(
        setup["train"]["dataset_dir"],
        should_load_dataset=args.load_dataset,
        should_save_dataset=args.save_dataset,
    )
    check_dataset_directory(
        setup["test"]["dataset_dir"],
        should_load_dataset=args.load_dataset_test,
        should_save_dataset=args.save_dataset_test,
    )
    return args, setup
