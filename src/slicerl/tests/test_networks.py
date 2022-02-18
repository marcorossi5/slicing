"""
    Ensures DUNEdn networks objects run the forwrd pass without errors.
"""
import logging
import numpy as np
from slicerl.networks.CMNet import CMNet
from slicerl.networks.HCNet import HCNet
from slicerl import PACKAGE

logger = logging.getLogger(PACKAGE + ".test")

supported_models = ["hcnet", "cmnet"]


def check_output_shape(model, oshape, eshape):
    """
    Parameters
    ----------
        - model: str, the model name
        - oshape: tuple, actual output shape
        - exhspae: tuple, expected output shape
    """
    if eshape != oshape:
        logger.info(
            f"{model} model expected shape is {eshape}, found {oshape}",
            extra={"status": "FAILED"},
        )
        raise
    else:
        logger.info(f"{model} forward pass", extra={"status": "PASSED"})


def run_test_cmnet():
    """Run test CM-Net."""
    # check that input and output have the same shape
    nb_hits = 100
    f_dims = 6
    x = np.random.randn(1, nb_hits, f_dims)
    kwargs = {
        "f_dims": f_dims,
        "nb_mha_heads": 3,
        "mha_filters": [12, 18],
        "nb_fc_heads": 3,
        "fc_filters": [8, 1],
        "name": "CM-Net",
    }
    cmnet = CMNet(**kwargs)
    output = cmnet(x)
    expected_shape = (1,)
    check_output_shape("CM-Net", output.shape, expected_shape)


def run_test_hcnet():
    """Run test HC-Net."""
    # check that input and output have the same shape
    nb_hits = 100
    f_dims = 6
    x = np.random.randn(1, nb_hits, f_dims)
    units = 4
    kwargs = {
        "f_dims": f_dims,
        "units": units,
        "nb_mha_heads": 3,
        "mha_filters": [12, 18],
        "nb_fc_heads": 3,
        "fc_filters": [8, 4],
        "name": "HC-Net",
    }
    cmnet = HCNet(**kwargs)
    output = cmnet(x)
    expected_shape = (1, nb_hits, units)
    check_output_shape("HC-Net", output.shape, expected_shape)


def run_test(modeltype):
    """
    Run the appropriate test for the supported model.
    Parameters
    ----------
        - modeltype: str, available options cmnet | hcnet
    """
    if modeltype == "cmnet":
        run_test_cmnet()
    elif modeltype == "hcnet":
        run_test_hcnet()


def test_networks():
    for modeltype in supported_models:
        run_test(modeltype)


if __name__ == "__main__":
    # format the test handler
    _console_handler = logging.StreamHandler()
    _console_handler.setLevel("INFO")
    _console_format = logging.Formatter("[%(status)s] (%(name)s) %(message)s")
    _console_handler.setFormatter(_console_format)
    logger.addHandler(_console_handler)
    logger.propagate = False
    test_networks()
