# This file is part of SliceRL by M. Rossi
from slicerl.config import NP_DTYPE
from slicerl.read_data import load_Events_from_file, load_Events_from_files

import tensorflow as tf
import numpy as np
from math import ceil

# ======================================================================
class EventDataset(tf.keras.utils.Sequence):
    """ Class defining dataset. """

    # ----------------------------------------------------------------------
    def __init__(self, data, batch_size, is_training=False, shuffle=True, **kwargs):
        """
        This generator must be used for training only.

        Parameters
        ----------
            - data    : list, network [inputs, targets]
            - is_training: bool, wether the dataset is used for training or
                           testing purposes
            - shuffle : bool, wether to shuffle dataset on epoch end
        """
        # needed to generate the dataset
        self.__events = data[0]
        self.nb_clusters_list = (
            [ev.nb_plane_clusters for ev in self.__events]
            if self.__events is not None
            else None
        )

        self.inputs = np.concatenate(data[1][0], axis=0) if is_training else data[1][0]
        self.targets = np.concatenate(data[1][1]) if is_training else data[1][1]
        self.batch_size = batch_size
        self.is_training = is_training
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.inputs))
        assert len(self.inputs) == len(
            self.targets
        ), f"Length of inputs and targets must match, got {len(self.inputs)} and {len(self.targets)}"
        if is_training:
            nb_positive = np.count_nonzero(self.targets)
            nb_all = len(self.targets)
            balancing = nb_positive / nb_all
            print(f"Training points: {nb_all} of which positives: {nb_positive}")
            print(f"Percentage of positives: {balancing}")
            self.balance_dataset(balancing)
        else:
            self.bal_inputs = self.inputs
            self.bal_targets = self.targets

    # ----------------------------------------------------------------------
    def balance_dataset(self, balancing):
        """
        Balances the dataset discarding part of the negative samples to match
        the size of the positive ones. This happens if the ratio of positve
        samples is below 20%.

        Parameters
        ----------
            - balancing: float, percentage of positive samples over the total
        """
        if balancing < 0.2:
            m_positives = self.targets.astype(bool)
            neg_idx = np.argwhere(~m_positives).flatten()
            neg_selected = np.random.choice(
                neg_idx, size=np.count_nonzero(m_positives), replace=False
            )
            m_negatives = np.isin(np.arange(len(self.targets)), neg_selected)
            mask = np.logical_or(m_positives, m_negatives)
            self.bal_inputs = self.inputs[mask]
            self.bal_targets = self.targets[mask]
        else:
            self.bal_inputs = self.inputs
            self.bal_targets = self.targets

        nb_positive = np.count_nonzero(self.bal_targets)
        nb_all = len(self.bal_targets)
        balancing = nb_positive / nb_all
        print("After balancing")
        print(f"Training points: {nb_all} of which positives: {nb_positive}")
        print(f"Percentage of positives: {balancing}")

    # ----------------------------------------------------------------------
    def on_epoch_end(self):
        if self.shuffle:
            perm = np.random.permutation(len(self.bal_inputs))
            self.bal_inputs = self.bal_inputs[perm]
            self.bal_targets = self.bal_targets[perm]

    # ----------------------------------------------------------------------
    def __getitem__(self, idx):
        batch_x = self.bal_inputs[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.bal_targets[idx * self.batch_size : (idx + 1) * self.batch_size]
        return batch_x, batch_y

    # ----------------------------------------------------------------------
    def __len__(self):
        return ceil(len(self.bal_inputs) / self.batch_size)

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
    nb_events = len(inputs)
    perm = np.random.permutation(nb_events)
    nb_split = int(split * nb_events)

    train_inp = [inputs[i] for i in perm[:nb_split]]
    train_trg = [targets[i] for i in perm[:nb_split]]

    val_inp = [inputs[i] for i in perm[nb_split:]]
    val_trg = [targets[i] for i in perm[nb_split:]]

    return [train_inp, train_trg], [val_inp, val_trg]


# ======================================================================
def generate_inputs_and_targets(event, is_training=False):
    """
    Takes an Event object and processes the 2D clusters to retrieve inputs and
    associated target arrays. Input arrays are concatenated cluster feacture
    vectors. Target arrays are binaries. If n is cluster number in an event, and
    is_training is False, n*(n-1)/2 clusters pairs are produced, n*(n-1) if
    is_training is True.

    Parameters
    ----------
        - event: Event, the event object holding the hits information
        - is_training: bool

    Returns
    -------
        - np.array, inputs of shape=(nb_clusters_pairs, nb_features*2)
        - np.array, target labels of shape=(nb_clusters_pairs)
    """
    # all clusters features
    acf = np.concatenate(
        [
            event.U.all_cluster_features,
            event.V.all_cluster_features,
            event.W.all_cluster_features,
        ],
        axis=0,
    )

    # cluster to main pfo
    c2mpfo = np.concatenate(
        [
            event.U.cluster_to_main_pfo,
            event.V.cluster_to_main_pfo,
            event.W.cluster_to_main_pfo,
        ],
        axis=0,
    )

    # create network inputs and targets
    inputs = []
    targets = []
    ped = 0
    for nb_cluster in event.nb_plane_clusters:
        for i in range(nb_cluster):
            for j in range(nb_cluster):
                if i == j:
                    continue
                if i < j and not is_training:
                    continue
                inps = np.concatenate([acf[ped + i], acf[ped + j]])
                inputs.append(inps)
                tgt = 1.0 if c2mpfo[ped + i] == c2mpfo[ped + j] else 0.0
                targets.append(tgt)
        ped += nb_cluster
    return np.stack(inputs, axis=0), np.array(targets)


# ======================================================================
def build_dataset(fn, nev=-1, min_hits=1, split=None, is_training=False):
    """
    Parameters
    ----------
        - fn       : list, of str events filenames
        - nev      : int, number of events to take from each file
        - min_hits : int, minimum hits per slice for dataset inclusion
        - split    : float, split percentage between train and val in events

    Returns
    -------
        - list, of Event instances
        - list, of inputs and targets couples:
            inputs is list of np.arrays of shape=(nb_cluster_pairs, nb_features);
            targets is a list of np.arrays of shape=(nb_cluster_pairs,)
    """
    if isinstance(fn, str):
        events = load_Events_from_file(fn, nev, min_hits)
    elif isinstance(fn, list) or isinstance(fn, tuple):
        events = load_Events_from_files(fn, nev, min_hits)
    else:
        raise NotImplementedError(f"please provide string or list, not {type(fn)}")

    inputs = []
    targets = []
    for event in events:
        inps, tgts = generate_inputs_and_targets(event, is_training=is_training)
        inputs.append(inps)
        targets.append(tgts)

    if split:
        train_splitted, val_splitted = split_dataset([inputs, targets], split)
        return [None, train_splitted], [None, val_splitted]
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
    batch_size = setup["model"]["batch_size"]

    train, val = build_dataset(
        fn, nev=nev, min_hits=min_hits, split=split, is_training=True
    )

    train_generator = EventDataset(
        train, is_training=True, batch_size=batch_size, shuffle=True
    )
    val_generator = (
        EventDataset(val, is_training=True, batch_size=batch_size) if split else None
    )
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
    batch_size = setup["model"]["batch_size"]

    data = build_dataset(fn, nev=nev, min_hits=min_hits)
    return EventDataset(data, batch_size=batch_size)


# ======================================================================
def dummy_dataset(nb_feats):
    """ Return a dummy dataset to build the model first when loading. """
    B = 32
    inputs = [np.random.rand(B, nb_feats) for i in range(2)]
    targets = [np.random.rand(B) for i in range(2)]
    data = (None, [inputs, targets])
    return EventDataset(data, batch_size=B, is_training=True)
