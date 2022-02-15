# This file is part of SliceRL by M. Rossi
""" This module contains utility functions of general interest. """
from pathlib import Path
import yaml
from hyperopt import hp

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
