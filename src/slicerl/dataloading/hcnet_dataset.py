import logging
from copy import deepcopy
import numpy as np
import tensorflow as tf
from slicerl import PACKAGE
from .read_data import load_events

logger = logging.getLogger(PACKAGE)


class HCEventDataset(tf.keras.utils.Sequence):
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
        self.y_pred = y_pred  # store the last predicted values
        if self.__events:
            for i, event in enumerate(self.__events):
                for j, plane in enumerate(event.planes):
                    y_sparse = y_pred.all_y_pred[3 * i + j]
                    plane.status = np.argmax(y_sparse, axis=1)
        else:
            raise ValueError(
                "Cannot set events attribute, found None"
                " (is the HCEventDataset generator in training mode?)"
            )


# ======================================================================
def augment_dataset(feats, nb_rotations):
    """
    Parameters
    ----------
        - feats: np.array, of shape=(nb coordinates, nb points)
        - nb_rotations: int, number of rotations to apply to inputs

    Returns
    -------
        - list: the augmented inputs
    """
    r = lambda t: np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    ts = np.linspace(0, 2 * np.pi, nb_rotations + 2)[:-1]
    rots = np.transpose(r(ts), [2, 0, 1])
    coords = np.einsum("ab,rac->rcb", feats[1:3], rots)
    energies = np.repeat(np.expand_dims(feats[:1], 0), nb_rotations + 1, axis=0)
    extras = np.repeat(np.expand_dims(feats[3:], 0), nb_rotations + 1, axis=0)
    new_feats = np.concatenate([energies, coords, extras], axis=1)
    return list(np.transpose(new_feats, [0, 2, 1]))


# ======================================================================
def _build_dataset(
    fn,
    nev,
    min_hits,
    should_split=False,
    split=None,
    should_standardize=None,
    nb_rotations=0,
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
        - should_standardize: bool, wether to standardize hit features plane by plane
        - nb_rotations: int, number of augmentations (hit coordinates rotations)

    Returns
    -------
        - tuple, of HCEventDataset objects if should_split is True. Single
          HCEventDataset object otherwise.
    """
    events = load_events(fn, nev, min_hits)

    # logging
    if should_standardize:
        logger.info("Standardizing input hit energies and cluster indices")
    if nb_rotations:
        logger.info(f"Augmenting dataset with {nb_rotations} rotations")

    inputs = []
    targets = []

    for ev in events:
        for plane in ev.planes:
            norm_clusters = plane.ordered_cluster_idx
            feats = deepcopy(plane.point_cloud)
            if should_standardize:
                feats[0] = (feats[0] - feats[0].mean()) / feats[0].std()
                norm_clusters = norm_clusters - norm_clusters.mean()
            inp = np.concatenate([plane.point_cloud, [norm_clusters]], axis=0)

            if nb_rotations:
                inputs.extend(augment_dataset(inp, nb_rotations))
                targets.extend([plane.ordered_mc_idx] * (nb_rotations + 1))
            else:
                inputs.append(inp.T)
                targets.append(plane.ordered_mc_idx)

    inputs = np.array(inputs, dtype=object)
    targets = np.array(targets, dtype=object)
    assert len(inputs) == len(
        targets
    ), f"Length of inputs and targets must match, got {len(inputs)} and {len(targets)}"

    # split dataset
    if should_split:
        logger.info(
            f"Splitting training dataset: {1-split:.2f} validation holdout percentage"
        )
        nb_events = len(inputs)
        perm = np.random.permutation(nb_events)
        nb_split = int(split * nb_events)

        train_inp = inputs[perm[:nb_split]]
        train_trg = targets[perm[:nb_split]]

        val_inp = inputs[perm[nb_split:]]
        val_trg = targets[perm[nb_split:]]

        # events are not returned when training
        return HCEventDataset([None, train_inp, train_trg]), HCEventDataset(
            [None, val_inp, val_trg]
        )
    return HCEventDataset([events, inputs, targets], shuffle=False)


# ======================================================================
def build_dataset(setup, is_training=None):
    key = "train" if is_training else "test"
    fn = setup[key]["fn"]
    nev = setup[key]["nev"]
    min_hits = setup[key]["min_hits"]
    split = setup["dataset"]["split"]
    should_split = is_training and (0 < split < 1)
    should_standardize = setup["dataset"]["standardize"]
    nb_rotations = setup["dataset"]["nb_rotations"] if is_training else 0
    return _build_dataset(
        fn,
        nev,
        min_hits,
        should_split=should_split,
        split=split,
        should_standardize=should_standardize,
        nb_rotations=nb_rotations,
    )
