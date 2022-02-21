# This file is part of SliceRL by M. Rossi
""" This module implements inference functions. """
from . import CMNet, HCNet


class Predictions:
    """Utility class to return RandLA-Net predictions."""

    def __init__(self, y_pred, preds=None, slices=None):
        """
        Parameters
        ----------
            - y_pred : list, of predicted vectors
            - preds  : list, of predicted adj matrices
            - slices : list, of sets
        """
        self.all_y_pred = y_pred
        self.all_events_preds = preds
        self.all_events_slices = slices

    # ----------------------------------------------------------------------
    def get_preds(self, index):
        """
        Returns the predictions for each possible edge in the i-th graph.
        Range is [0,1].

        Parameters
        ----------
            - index : int, index in preds list

        Returns
        -------
            - np.array: status at index i of shape=(N,N)
        """
        return self.all_events_preds[index]

    # ----------------------------------------------------------------------
    def get_slices(self, index):
        """
        Returns the slices: each sl contains the cluster indices inside the
        sl set.

        Parameters
        ----------
            - index : int, index in sl list

        Returns
        -------
            - list: of set objects with calohit indices
        """
        return self.all_events_slices[index]


# ======================================================================
def get_prediction(network, test_generator, batch_size, threshold=0.5):
    """Inference Wrapper function."""
    if isinstance(network, CMNet.CMNet):
        outputs = CMNet.inference(network, test_generator, batch_size, threshold)
    elif isinstance(network, HCNet.HCNet):
        outputs = HCNet.inference(network, test_generator, batch_size)
    else:
        raise NotImplementedError(f"Model not implemented, got {type(network)}")
    return Predictions(outputs)
