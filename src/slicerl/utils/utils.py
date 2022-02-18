# This file is part of SliceRL by M. Rossi
""" This module contains utility functions of general interest. """
import shutil
import logging
import argparse
from pathlib import Path
from hyperopt import hp
import yaml
from slicerl import PACKAGE

logger = logging.getLogger(PACKAGE)


def makedir(folder):
    """Create directory."""
    if not folder.exists():
        folder.mkdir()
    else:
        raise Exception(f"Output folder {folder} already exists.")


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
def save_runcard(fname, setup, modify=True):
    with open(fname, "w") as f:
        # yaml is not able to save the Path objects
        # TODO: overload the yaml class
        if modify:
            setup["output"] = setup["output"].as_posix()
            setup["train"]["dataset_dir"] = setup["train"]["dataset_dir"].as_posix()
            setup["test"]["dataset_dir"] = setup["test"]["dataset_dir"].as_posix()
        yaml.dump(setup, f, indent=4)
        modify_runcard(setup)


# ======================================================================
def modify_runcard(setup):
    """
    Loads correctly the Path objects in the runcard.

    Parameters
    ----------
        - setup: dict, the loaded settings
    """
    setup.update({"output": Path(setup["output"])})
    setup["train"].update({"dataset_dir": Path(setup["train"]["dataset_dir"])})
    setup["test"].update({"dataset_dir": Path(setup["test"]["dataset_dir"])})


# ======================================================================
# argument parsing function
def get_cmd_args():
    """
    Main parsing function.

    Returns
    -------
        - argparse.NameSpace, the command line arguments

    """
    parser = argparse.ArgumentParser(
        """
    Cluster Merging Network for slicing
    """
    )

    # main options
    parser.add_argument(
        "runcard",
        action="store",
        nargs="?",
        type=Path,
        default=None,
        help="A yaml file with the setup",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output folder",
        type=Path,
        default=Path("default_output"),
    )
    parser.add_argument(
        "--model", "-m", type=Path, default=None, help="The input model folder"
    )

    # boolean flags
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
        "--no_graphics",
        help="DO not display figures via GUI",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--just_train",
        action="store_true",
        help="Just train with ../dataset/train folder",
    )
    return parser.parse_args()


# ======================================================================
# argument checking functions
def check_cmd_args(args):
    """
    Check that the command line arguments are coherent.

    Parameters
    ----------
        - args: argparse.Namespace, the command line arguments

    Raises
    ------
        - ValueError, if namespace contains both `runcard` and `model`, or if
                      both save and load dataset options are specified
        - FileExistsError, if `runcard` is not a valid file
    """
    if (not args.model and not args.runcard) or (args.model and args.runcard):
        raise ValueError("Invalid options: requires either input runcard or model.")
    elif args.runcard and not args.runcard.is_file():
        raise FileExistsError("Invalid runcard: not a file.")
    if args.force:
        logger.warning("Running with --force option will overwrite existing model")
    if (args.save_dataset and args.load_dataset) or (
        args.save_dataset_test and args.load_dataset_test
    ):
        raise ValueError("Invalid options: requires either save or load dataset.")


# ======================================================================
def initialize_output_folder(output, should_force, should_scan):
    """
    Parameters
    ----------
        - output: Path, the ouptut directory
        - should_force: bool, wether to replace the already existing output directory
        - should_scan: bool, wether to create directory for hyperparameter scan
    """
    try:
        makedir(output)
        makedir(output / "plots")
        if should_scan:
            makedir(output / "hopt")
    except Exception as error:
        if should_force:
            logger.warning(f"Overwriting {output} with new model")
            shutil.rmtree(output)
            makedir(output)
            makedir(output / "plots")
            if should_scan:
                makedir(output / "hopt")
        else:
            logger.error(error)
            logger.error('Delete or run with "--force" to overwrite.')
            exit(-1)


# ======================================================================
def check_dataset_directory(
    dataset_dir, should_load_dataset=False, should_save_dataset=False
):
    """
    If should_load_dataset is true, checks if dataset directory exists and
    raises error if not. If should_save_dataset is true, checks if dataset
    directory exists and creates it. Raises error if parent folder does not exist.

    Parameters
    ----------
        - dataset_dir: Path
        - should_load_dataset: bool
        - should_save_dataset: bool

    Raises
    ------
        - FileNotFoundError if directory does not exist
    """
    if should_load_dataset and not dataset_dir.exists():
        raise FileNotFoundError(f"no such file or directory: {dataset_dir}")

    if should_save_dataset and not dataset_dir.exists():
        dataset_dir.mkdir()
