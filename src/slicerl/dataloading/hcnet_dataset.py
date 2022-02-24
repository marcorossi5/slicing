import logging
from copy import deepcopy
import numpy as np
import tensorflow as tf
from slicerl import PACKAGE
from slicerl.utils.tools import onehot
from .read_data import load_events

logger = logging.getLogger(PACKAGE)


class EventDataset(tf.keras.utils.Sequence):
    def __init__(
        self,
        inputs,
        shuffle=False,
        seed=12345,
    ):
        self.__events, self.inputs, self.targets = inputs
        self.shuffle = shuffle
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.perm = np.arange(self.__len__())

    # ----------------------------------------------------------------------
    def on_epoch_end(self):
        if self.shuffle:
            self.perm = self.rng.permutation(self.__len__())

    # ----------------------------------------------------------------------
    def __getitem__(self, idx):
        ii = self.perm[idx]
        batch_x = self.inputs[ii][None]
        batch_y = self.targets[ii][None]
        return batch_x, batch_y

    # ----------------------------------------------------------------------
    def __len__(self):
        return len(self.inputs)

    # ----------------------------------------------------------------------
    @property
    def events(self):
        return self.__events

    # ----------------------------------------------------------------------
    @events.setter
    def events(self, y_pred):
        """
        Parameters
        ----------
            - y_pred: Predictions, object storing network predictions
        """
        logger.info("Setting events")
        if self.__events:
            for i, event in enumerate(self.__events):
                for j, plane in enumerate(event.planes):
                    y_sparse = y_pred.all_y_pred[3 * i + j]
                    plane.status = np.argmax(y_sparse, axis=1)

                    # just a debugging plot
                    # import matplotlib.pyplot as plt
                    # plt.subplot(2,1,1)
                    # plt.plot(y_sparse[:30,].T, lw=0.5)
                    # idx = np.argmax(y_sparse,axis=1)
                    # bins = np.linspace(-0.5, 63.5, 65)
                    # h, _ = np.histogram(idx, bins=bins)
                    # plt.subplot(2,1,2)
                    # plt.hist(bins[:-1], bins, weights=h)
                    # plt.show()
        else:
            raise ValueError(
                "Cannot set events attribute, found None"
                " (is the EventDataset generator in training mode?)"
            )


# ======================================================================
def _build_dataset(
    fn, nev, min_hits, should_split=False, split=None, should_standardize=None
):
    """
    Parameters
    ----------
        - fn: list, events file names
        - nev: int, number of events to take from each file
        - min_hits: int, minimum hits per slice for dataset inclusion
        - depth: int, depth of targets one hot encoding
        - should_split: bool, wether to hold out validation set
        - split: float, split percentage in the [0,1] range
        - should_standardize: standardize hit features plane by plane

    Returns
    -------
        - tuple, of EventDataset objects if should_split is True. Single
          EventDataset object otherwise.
    """
    events = load_events(fn, nev, min_hits)

    inputs = []
    targets = []

    for ev in events:
        for plane in ev.planes:
            norm_clusters = plane.ordered_cluster_idx
            feats = deepcopy(plane.point_cloud)
            if should_standardize:
                feats[0] = (feats[0] - feats[0].mean()) / feats[0].std()
                feats[1:3] = feats[1:3] - feats[1:3].mean(1, keepdims=True)
                norm_clusters = (norm_clusters - norm_clusters.mean())

            # norm = len(set(plane.ordered_cluster_idx))
            # norm_clusters = plane.ordered_cluster_idx / norm
            inputs.append(
                np.concatenate([plane.point_cloud, [norm_clusters]], axis=0).T
            )
            targets.append(plane.ordered_mc_idx)

    inputs = np.array(inputs, dtype=object)
    targets = np.array(targets, dtype=object)
    assert len(inputs) == len(
        targets
    ), f"Length of inputs and targets must match, got {len(inputs)} and {len(targets)}"

    # split dataset
    if should_split:
        nb_events = len(inputs)
        perm = np.random.permutation(nb_events)
        nb_split = int(split * nb_events)

        train_inp = inputs[perm[:nb_split]]
        train_trg = targets[perm[:nb_split]]

        val_inp = inputs[perm[nb_split:]]
        val_trg = targets[perm[nb_split:]]

        # events are not returned when training
        return EventDataset([None, train_inp, train_trg]), EventDataset(
            [None, val_inp, val_trg]
        )
    return EventDataset([events, inputs, targets], shuffle=False)


# ======================================================================
def build_dataset(setup, is_training=None):
    key = "train" if is_training else "test"
    fn = setup[key]["fn"]
    nev = setup[key]["nev"]
    min_hits = setup[key]["min_hits"]
    split = setup["dataset"]["split"]
    should_split = is_training and (0 < split < 1)
    should_standardize = setup["dataset"]["standardize"]
    return _build_dataset(
        fn,
        nev,
        min_hits,
        should_split=should_split,
        split=split,
        should_standardize=should_standardize,
    )
