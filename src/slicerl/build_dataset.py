# This file is part of SliceRL by M. Rossi
from slicerl.config import NP_DTYPE
from slicerl.read_data import load_Events_from_file, load_Events_from_files
from slicerl.tools import onehot, onehot_to_indices
from slicerl.layers import LocSE
from slicerl.elliptic_topK import EllipticTopK

import tensorflow as tf
import numpy as np
import knn

# ======================================================================
class EventDataset(tf.keras.utils.Sequence):
    """ Class defining dataset. """

    # ----------------------------------------------------------------------
    def __init__(self, data, shuffle=True, **kwargs):
        """
        Parameters
        ----------
            - data    : list, [inputs, targets] for RandLA-Net
            - shuffle : bool, wether to shuffle dataset on epoch end
        """
        # needed to generate the dataset
        self.__events = data[0]
        self.inputs, self.targets = data[1]
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.inputs))
        assert len(self.inputs) == len(
            self.targets
        ), f"Length of inputs and targets must match, got {len(self.inputs)} and {len(self.targets)}"

    # ----------------------------------------------------------------------
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    # ----------------------------------------------------------------------
    def __getitem__(self, idx):
        index = self.indexes[idx]
        return self.inputs[index], self.targets[index]

    # ----------------------------------------------------------------------
    def __len__(self):
        return len(self.inputs)

    # ----------------------------------------------------------------------
    def get_pc(self, index):
        """
        Get the point cloud at a certain index.
        TODO: this has to be changed into cluster input adj or similar

        Returns
        -------
            - np.array, point cloud of shape=(N,2)
        """
        return self.events[index].point_cloud[1:3].T

    # ----------------------------------------------------------------------
    def get_status(self, index):
        """
        Returns the status vector of event at index `index`.

        Returns
        -------
            - np.array, point cloud of shape=(N,)
        """
        return self.events[index].status

    # ----------------------------------------------------------------------
    def get_targets(self, index):
        """
        Returns the mc status vector of event at index `index`.

        Returns
        -------
            - np.array, point cloud of shape=(N,)
        """
        return self.events[index].ordered_mc_idx

    # ----------------------------------------------------------------------
    @property
    def events(self):
        """ Event attribute. """
        return self.__events

    # ----------------------------------------------------------------------
    @events.setter
    def events(self, y_pred):
        """
        Parameters
        ----------
            - y_pred: Predictions, object storing network predictions
        """
        print("[+] setting events")
        if self.__events:
            for i, event in enumerate(self.__events):
                event.store_preds(y_pred.get_slices(i))
        else:
            raise ValueError(
                "Cannot set events attribute, found None"
                " (is the EventDataset generator in training mode?)"
            )


# ======================================================================
def rotate_pc(pc, t):
    """
    Augment points `pc` with plane rotations given by a list of angles `t`.

    Parameters
    ----------
        - pc : tf.Tensor, point cloud of shape=(N,2)
        - t  : list, list of angles in radiants of length=(B,)

    Returns
    -------
        tf.Tensor, batch of rotated point clouds of shape=(B,N,2)
    """
    # rotation matrices
    sint = np.sin(t)
    cost = np.cos(t)
    irots = np.array([[cost, sint], [-sint, cost]])
    rots = np.moveaxis(irots, [0, 1, 2], [2, 1, 0])
    return np.matmul(pc, rots)


# ======================================================================
def transform(pc, feats, target):
    """
    Augment a inputs rotating points. Targets and feats remain the same

    Parameters
    ----------
        - pc      : tf.Tensor, point cloud of shape=(N,2)
        - feats   : tf.Tensor, feature point cloud of shape=(N,2)
        - target : tf.Tensor, segmented point cloud of shape=(N,)

    Returns
    -------
        - tf.Tensor, augmented point cloud of shape=(B,N,2)
        - tf.Tensor, repeated feature point cloud on batch axis of shape=(B,N,2)
        - tf.Tensor, repeated segmented point cloud on batch axis of shape=(B,N)
    """
    # define rotating angles list
    fracs = np.array(
        [
            0,
            0.16,
            0.25,
            0.3,
            0.5,
            0.66,
            0.75,
            0.83,
            1,
            -0.16,
            -0.25,
            -0.3,
            -0.5,
            -0.66,
            -0.75,
            -0.83,
        ]
    )
    t = np.pi * fracs
    # random permute the angles list to ensure no bias
    randt = np.random.permutation(t)
    B = len(t)

    # produce augmented pc
    pc = rotate_pc(pc, randt)

    # repeat feats and target along batch axis
    feats = np.repeat(feats[None], B, axis=0)
    target = np.repeat(target[None], B, axis=0)

    return pc, feats, target


# ======================================================================
def split_dataset(data, split=0.5):
    """
    Parameters
    ----------
        - data  : list, [inputs, targets] for RandLA-Net to split
        - split : list, [validation, test] percentages

    Returns
    -------
        - list, [inputs, targets] for validation
        - list, [inputs, targets] for testing
    """
    inputs, targets = data
    assert len(inputs) == len(
        targets
    ), f"Length of inputs and targets must match, got {len(inputs)} and {len(targets)}"
    l = len(inputs)
    split_idx = int(l * split)

    val_inputs = inputs[:split_idx]
    test_inputs = inputs[split_idx:]

    val_targets = targets[:split_idx]
    test_targets = targets[split_idx:]

    return [val_inputs, val_targets], [test_inputs, test_targets]


# ======================================================================
def build_dataset(fn, nev=-1, min_hits=1, split=None):
    """
    Parameters
    ----------
        - fn         : list, of str events filenames
        - nev        : int, number of events to take from each file
        - min_hits   : int, minimum hits per slice for dataset inclusion
        - split      : float, split percentage between train and val in events

    Returns
    -------
        - list, of Event instances
        - list, of inputs and targets couples:
            inputs is list of np.arrays [1, nb_cluster, nb_cluster, 2, nb_features];
            targets is a list of np.arrays of shape=(1,nb_cluster,nb_cluster)
    """
    if isinstance(fn, str):
        events = load_Events_from_file(fn, nev, min_hits)
    elif isinstance(fn, list) or isinstance(fn, tuple):
        events = load_Events_from_files(fn, nev, min_hits)
    else:
        raise NotImplementedError(
            f"please provide string or list, not {type(fn)}"
        )
    inputs = [event.cluster_features[None] for event in events]
    targets = [event.target_adj[None] for event in events]

    if split:
        nb_events = len(inputs)
        perm = np.random.permutation(nb_events)
        nb_split = int(split * nb_events)

        train_evt = None
        train_inp = [inputs[i] for i in perm[:nb_split]]
        train_trg = [targets[i] for i in perm[:nb_split]]

        val_evt = None
        val_inp = [inputs[i] for i in perm[nb_split:]]
        val_trg = [targets[i] for i in perm[nb_split:]]

        return (
            [train_evt, [train_inp, train_trg]],
            [val_evt, [val_inp, val_trg]],
        )
    return events, [inputs, targets]


# ======================================================================
def build_dataset_train(setup):
    """
    Wrapper function to build dataset for training. Implements validation
    splitting according to config file.

    Returns
    -------
        - EventDataset, object for training
        - EventDataset, object for validation (None if split percentage is None)
    """
    fn = setup["train"]["fn"]
    nev = setup["train"]["nev"]
    min_hits = setup["train"]["min_hits"]
    split = setup["dataset"]["split"]

    train, val = build_dataset(
        fn,
        nev=nev,
        min_hits=min_hits,
        split=split,
    )
    train_generator = EventDataset(train, shuffle=True)
    val_generator = EventDataset(val) if split else None
    return train_generator, val_generator


# ======================================================================
def build_dataset_test(setup):
    """
    Wrapper function to build dataset for testing.

    Returns
    -------
        - EventDataset, object for testing
    """
    fn = setup["test"]["fn"]
    nev = setup["test"]["nev"]
    min_hits = setup["test"]["min_hits"]

    data = build_dataset(fn, nev=nev, min_hits=min_hits)
    return EventDataset(data)


# ======================================================================
def dummy_dataset(nb_feats):
    """ Return a dummy dataset to build the model first when loading. """
    B = 1
    N = 50
    inputs = np.random.rand(B, N, N, 2, nb_feats)
    targets = np.random.rand(B, N, N)
    data = (None, [inputs, targets])
    return EventDataset(data)
