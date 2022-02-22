"""
    This module reads the command line options and initializes the directory
    structure.
"""
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
    Set the host device for tensorflow.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) == 0:
        return
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    gpus = setup.get("gpu")
    if gpus is not None:
        if isinstance(gpus, int):
            gpus = [gpus]
        gpus = [
            tf.config.PhysicalDevice(f"/physical_device:GPU:{gpu}", "GPU")
            for gpu in gpus
        ]
        tf.config.set_visible_devices(gpus, "GPU")
        logger.warning(f"Host device: GPU {gpus}")
    else:
        logger.warning("Host device: CPU")

    if setup.get("debug"):
        logger.warning("Run all tf functions eagerly")
        tf.config.run_functions_eagerly(True)


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
        save_runcard(args.output / "runcard.yaml", setup, modify=False)
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

    from .configflow import set_manual_seed

    set_manual_seed(args.seed)
    return args, setup
