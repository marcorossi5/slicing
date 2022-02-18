# This file is part of SliceRL by M. Rossi
from operator import is_
from . import cmnet_dataset as cmset
from . import hcnet_dataset as hcset


def build_dataset(
    setup,
    from_np_path=None,
    is_training=None,
    should_save_dataset=None,
    should_load_dataset=None,
):
    """Wrapper function."""
    modeltype = setup["model"]["net_type"]
    if modeltype == "CM":
        if from_np_path is not None:
            return cmset.build_dataset_from_np(setup, from_np_path)
        if is_training:
            return cmset.build_dataset_train(setup, should_save_dataset, should_load_dataset)
        else:
            return cmset.build_dataset_test(setup, should_save_dataset, should_load_dataset)
    elif modeltype == "HC":
        return hcset.build_dataset(setup, is_training=is_training)
    else:
        raise NotImplementedError(f"Model not implemented, got {modeltype}")
