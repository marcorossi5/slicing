# This file is part of SliceRL by M. Rossi
from slicerl.config import NP_DTYPE
from slicerl.read_data import load_Events_from_file, load_Events_from_files

import tensorflow as tf
import numpy as np
from math import ceil

plane_to_idx = {
    "U": 0,
    "V": 1,
    "W": 2
}

# ======================================================================
class EventDataset(tf.keras.utils.Sequence):
    """ Class defining dataset. """

    # ----------------------------------------------------------------------
    def __init__(
        self,
        data,
        batch_size,
        cthresholds=None,
        is_training=False,
        shuffle=False,
        verbose=1,
    ):
        """
        This generator must be used for training only.

        Parameters
        ----------
            - data: list, network [event, [inputs, targets]]
            - batch_size: int, batch size
            - cthresholds: list, for each plane view the index in the adjacency
                           matrix that separates the two types of clusters
            - is_training: bool, wether the dataset is used for training or
                           testing purposes
            - shuffle: bool, wether to shuffle dataset on epoch end
            - verbose: int, print balancing stats
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
        self.cthresholds = cthresholds
        self.is_training = is_training
        self.shuffle = shuffle
        self.verbose = verbose

        self.indexes = np.arange(len(self.inputs))
        assert len(self.inputs) == len(
            self.targets
        ), f"Length of inputs and targets must match, got {len(self.inputs)} and {len(self.targets)}"
        if is_training:
            nb_positive = np.count_nonzero(self.targets)
            nb_all = len(self.targets)
            balancing = nb_positive / nb_all
            if self.verbose:
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

        if self.verbose:
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
def generate_inputs_and_targets(event, is_training=False, min_hits=1, plane=None):
    """
    Takes an Event object and processes the 2D clusters to retrieve inputs and
    associated target arrays. Input arrays are concatenated cluster feacture
    vectors. Target arrays are binaries. If n is cluster number in an event, and
    is_training is False, n*(n-1)/2 clusters pairs are produced, n*(n-1) if
    is_training is True.
    Filters the clusters by number of hits.

    Parameters
    ----------
        - event: Event, the event object holding the hits information
        - is_training: bool
        - plane: int, plane index to train network only on specific plane

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
    sl = slice(plane, plane + 1) if is_training else slice(None)
    zipped = zip(event.nb_plane_clusters[sl], event.nb_plane_hits[sl])
    for nb_cluster, nb_hits in zipped:
        for i in range(nb_cluster):
            for j in range(nb_cluster):
                # filtering
                if i == j:
                    continue
                if i < j and not is_training:
                    continue
                nb_hits1 = np.ceil(acf[ped + i, 0] * nb_hits)
                nb_hits2 = np.ceil(acf[ped + j, 0] * nb_hits)
                if nb_hits1 < min_hits or nb_hits2 < min_hits:
                    continue
                # get inputs
                cf0 = acf[ped + i]
                cf1 = acf[ped + j]
                sdot = np.abs((cf0[3:5] * cf1[3:5]).sum())
                inps = np.concatenate([cf0, cf1, [sdot]])
                inputs.append(inps)
                # get targets
                tgt = 1.0 if c2mpfo[ped + i] == c2mpfo[ped + j] else 0.0
                targets.append(tgt)
        ped += nb_cluster
    return np.stack(inputs, axis=0), np.array(targets)


# ======================================================================
def load_events(fn, nev, min_hits):
    """Loads event from file into a list of Event objects. Supported inputs are
    either a single file name or a list of file names.

    Parameters
    ----------
        - fn: str or list, events file names
        - nev: int, number of events to take from each file
        - min_hits : int, minimum hits per slice for dataset inclusion

    Returns
    -------
        - list of Event objects

    Raises
    ------
        - NotImplementedError if fn is not str, list or tuple
    """
    if isinstance(fn, str):
        events = load_Events_from_file(fn, nev, min_hits)
    elif isinstance(fn, list) or isinstance(fn, tuple):
        events = load_Events_from_files(fn, nev, min_hits)
    else:
        raise NotImplementedError(f"please provide string or list, not {type(fn)}")
    return events


# ======================================================================
def get_generator(
    events, batch_size, split=False, is_training=False, min_hits=1, cthreshold=None, plane=None
):
    """
    Wrapper function to obtain an event dataset ready for inference directly
    from a list of events.

    Parameters
    ----------
        - events: list, of Event instances
        - batch_size: int, batch size
        - split: float, split percentage between train and val in events
        - is_training: bool, wether the dataset is used for training or
                       testing purposes
        - min_hits : int, minimum hits per input cluster for dataset inclusion
        - cthreshold: int, size of input cluster above which a cluster is
                      considered to be large
        - plane: int, plane index to train network only on specific plane

    Returns
    -------
        - EventDataset, object for inference
    """
    kwargs = {"batch_size": batch_size, "is_training": is_training, "shuffle": False}
    inputs = []
    targets = []
    cthresholds = []
    for event in events:
        event.refine()
        inps, tgts = generate_inputs_and_targets(
            event, is_training=is_training, min_hits=min_hits, plane=plane
        )
        if cthreshold is not None:
            cthresholds.append(get_cluster_thresholds(event, cthreshold))
        inputs.append(inps)
        targets.append(tgts)
    if split:
        train_splitted, val_splitted = split_dataset([inputs, targets], split)
        train_data = [None, train_splitted]
        val_data = [None, val_splitted]
        keys = kwargs.copy()
        keys.update(shuffle=True)
        return EventDataset(train_data, **keys), EventDataset(val_data, **kwargs)
    data = events, [inputs, targets]
    return EventDataset(data, cthresholds=cthresholds, **kwargs)


# ======================================================================
def build_dataset(
    fn, batch_size, nev=-1, min_hits=1, split=None, is_training=False, cthreshold=None, plane=None
):
    """
    Loads first the events from file, then generates the dataset to be fed into
    FFNN.

    Parameters
    ----------
        - fn: list, of str events filenames
        - nev: int, number of events to take from each file
        - min_hits: int, minimum hits per input cluster for dataset inclusion
        - split: float, split percentage between train and val in events
        - plane: int, plane index to train network only on specific plane

    Returns
    -------
        - list, of Event instances
        - list, of inputs and targets couples:
            inputs is list of np.arrays of shape=(nb_cluster_pairs, nb_features);
            targets is a list of np.arrays of shape=(nb_cluster_pairs,)
    """
    events = load_events(fn, nev, min_hits)
    return get_generator(
        events,
        batch_size,
        split=split,
        is_training=is_training,
        min_hits=min_hits,
        cthreshold=cthreshold,
        plane=plane
    )


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
    plane = plane_to_idx[setup["train"]["plane"]]

    return build_dataset(
        fn, batch_size, nev=nev, min_hits=min_hits, split=split, is_training=True, plane=plane
    )


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
    cthreshold = setup["test"]["cthreshold"]

    return build_dataset(
        fn, batch_size, nev=nev, min_hits=min_hits, cthreshold=cthreshold
    )


# ======================================================================
def dummy_dataset(nb_feats):
    """ Return a dummy dataset to build the model first when loading. """
    B = 32
    inputs = [np.random.rand(B, nb_feats) for i in range(2)]
    targets = [np.random.rand(B) for i in range(2)]
    data = (None, [inputs, targets])
    return EventDataset(data, batch_size=B, is_training=True)


# ======================================================================
def get_cluster_thresholds(event, cthreshold):
    """
    Parameters
    ----------
        - event: Event instance
        - cthreshold: list, size of input cluster above which a cluster is
                      considered to be large

    Returns
    -------
        - list, for each plane view in the event, the index in the adjacency matrix that
          separates the two types of clusters
    """
    cthresholds = []
    for plane in event.planes:
        plane_thresholds = []
        map_fn = lambda x: np.count_nonzero(plane.status == x)
        csize = np.array(list(map(map_fn, sorted(plane.cluster_set))))
        for ct in cthreshold:
            plane_thresholds.append(np.count_nonzero(csize > ct))
        cthresholds.append(plane_thresholds)
    return cthresholds
