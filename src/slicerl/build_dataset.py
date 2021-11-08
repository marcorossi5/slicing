# This file is part of SliceRL by M. Rossi
from threading import currentThread
from slicerl.config import NP_DTYPE
from slicerl.read_data import load_Events_from_file, load_Events_from_files
from slicerl.Event import get_cluster_features

import tensorflow as tf
import numpy as np
from math import ceil

plane_to_idx = {"U": 0, "V": 1, "W": 2}

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
        verbose=0,
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
        self.c_indices = np.concatenate(data[1][2]) if is_training else data[1][2]
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
            # no balance needed for cluster indices because they enter inference only

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
        - data  : list, [inputs, targets, c indices] for RandLA-Net to split
        - split : list, [validation, test] percentages

    Returns
    -------
        - list, [inputs, targets] for validation
        - list, [inputs, targets] for testing
    """
    inputs, targets, c_indices = data
    assert len(inputs) == len(
        targets
    ), f"Length of inputs and targets must match, got {len(inputs)} and {len(targets)}"
    nb_events = len(inputs)
    perm = np.random.permutation(nb_events)
    nb_split = int(split * nb_events)

    train_inp = [inputs[i] for i in perm[:nb_split]]
    train_trg = [targets[i] for i in perm[:nb_split]]
    train_c = [c_indices[i] for i in perm[:nb_split]]

    val_inp = [inputs[i] for i in perm[nb_split:]]
    val_trg = [targets[i] for i in perm[nb_split:]]
    val_c = [c_indices[i] for i in perm[nb_split:]]

    return [train_inp, train_trg, train_c], [val_inp, val_trg, val_c]


# ======================================================================
def generate_inputs_and_targets(event, min_hits=1):
    """
    Takes an Event object and processes the 2D clusters to retrieve inputs and
    associated target arrays. Input arrays are concatenated cluster feacture
    vectors. Target arrays are binaries. If n is cluster number in an event,
    n*(n-1)/2 clusters pairs are produced. Filters the clusters by number of hits.

    Parameters
    ----------
        - event: Event, the event object holding the hits information
        - is_training: bool

    Returns
    -------
        - np.array, inputs of shape=(nb_clusters_pairs, nb_features*2)
        - np.array, target labels of shape=(nb_clusters_pairs)
    """
    # create network inputs and targets
    inputs = []
    targets = []
    c_indices = []
    ped = 0

    for plane in event.planes:
        for i in range(plane.nb_clusters):
            for j in range(i):
                icluster = plane.state(i)
                jcluster = plane.state(j)

                if icluster.shape[1] < min_hits or icluster.shape[1] < min_hits:
                    # TODO: is this needed?
                    continue

                # find intra-cluster properties
                ifeats = plane.all_cluster_features[i]
                jfeats = plane.all_cluster_features[j]

                # find intra-merged cluster properties
                mcluster = np.concatenate([icluster, jcluster], axis=1)
                mfeats = get_cluster_features(
                    mcluster, len(mcluster) / len(plane), plane.tpc_view
                )

                # add some inter-cluster properties
                # min and max inter-cluster distance
                idelim = ifeats[15:23].reshape(2, 4).T
                jdelim = jfeats[15:23].reshape(2, 4).T
                min_dist = np.inf
                for ip in idelim:
                    for jp in jdelim:
                        dist = ((ip - jp) ** 2).sum()
                        if dist < min_dist:
                            min_dist = dist
                sqmin_dist = np.sqrt(min_dist)

                # |abs(cos(t))|, t angle between expected directions
                sdot = np.abs((ifeats[3:5] * jfeats[3:5]).sum())

                inputs.append(
                    np.concatenate(
                        [
                            ifeats,
                            jfeats,
                            mfeats,
                            [sdot],
                            [sqmin_dist],
                        ],
                        axis=0,
                    )
                )

                # targets
                ipfo = plane.cluster_to_main_pfo[i]
                jpfo = plane.cluster_to_main_pfo[j]
                tgt = 1.0 if ipfo == jpfo else 0.0
                targets.append(tgt)

                # indices in the adj matrix
                c_indices.append([ped + i, ped + j])
        ped += plane.nb_clusters
    return np.stack(inputs, axis=0), np.array(targets), np.array(c_indices)


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
    events, inputs, targets, c_indices, cthresholds, batch_size, is_training, split
):
    """
    Get EventDataset generator from data.
    """
    kwargs = {"batch_size": batch_size, "is_training": is_training, "shuffle": False}
    if split:
        train_splitted, val_splitted = split_dataset(
            [inputs, targets, c_indices], split
        )
        train_data = [None, train_splitted]
        val_data = [None, val_splitted]
        keys = kwargs.copy()
        keys.update(shuffle=True)
        return EventDataset(train_data, **keys), EventDataset(val_data, **kwargs)
    data = events, [inputs, targets, c_indices]
    return EventDataset(data, cthresholds=cthresholds, **kwargs)


# ======================================================================
def generate_dataset(
    events, min_hits=1, cthreshold=None, should_save_dataset=False, dataset_dir=None
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
        - should_save_dataset: bool, wether to to save the dataset in directory
                               specified by runcard
        - dataset_dir: Path, dataset directory

    Returns
    -------
        - list: network inputs, of length=(nb_events)
                each element is a np.array of shape=(nb_cluster_pairs, nb_feats)
        - list: network targets, of length=(nb_events)
                each element is a np.array of shape=(nb_cluster_pairs,)
        - list: adj  cluster pair indices, of length=(nb_events)
                each element is a np.array of shape=(nb_cluster_pairs, 2)
        - list: cluster thresholds, of length=(nb_events).
                For inference only: cluster indices after which clusters should
                be considered small. Each element is of shape=(3, nb_threhsolds),
                default nb_threhsolds is 2.
    """
    inputs = []
    targets = []
    c_indices = []
    cthresholds = []
    for event in events:
        event.refine()
        inps, tgts, c_idxs = generate_inputs_and_targets(event, min_hits=min_hits)
        if cthreshold is not None:
            cthresholds.append(get_cluster_thresholds(event, cthreshold))
        inputs.append(inps)
        targets.append(tgts)
        c_indices.append(c_idxs)
    if should_save_dataset:
        save_dataset(dataset_dir, inputs, targets, c_indices, cthresholds)
    return inputs, targets, c_indices, cthresholds


# ======================================================================
def load_dataset(dataset_dir):
    """
    Load dataset from directory.

    Parameters
    ----------
        - dataset_dir: Path, directory where the dataset is stored

    Returns
    -------
        - list, network inputs
        - list, network targets
        - list, adj matrix indices
        - list, cluster thresholds for inference
    """
    row_splits = np.load(dataset_dir / "event_row_splits.npy")
    splittings = np.cumsum(row_splits[:-1])
    split_fn = lambda x: np.split(x, splittings, axis=0)

    # load files
    inputs = np.load(dataset_dir / "inputs.npy")
    targets = np.load(dataset_dir / "targets.npy")
    c_indices = np.load(dataset_dir / "c_indices.npy")
    fname = dataset_dir / "cthresholds.npy"
    cthresholds = split_fn(np.load(fname)) if fname.is_file() else []

    # check if inputs are consistents
    assert sum(row_splits) == len(
        inputs
    ), f"Dataset loading failes: input arrays first axis lengths should match, found {sum(row_splits)} and {len(inputs)}"
    assert len(inputs) == len(targets)
    return split_fn(inputs), split_fn(targets), split_fn(c_indices), cthresholds


# ======================================================================
def save_dataset(dataset_dir, inputs, targets, c_indices, cthresholds):
    """
    Save dataset in directory.

    Parameters
    ----------
        - dataset_dir: Path, directory where the dataset is stored
        - inputs: list
        - targets: list
        - c_indices: list, adj matrix indices
        - cthresholds: list
    """
    row_splits = np.array([len(tgt) for tgt in targets])
    np.save(dataset_dir / "inputs.npy", np.concatenate(inputs))
    np.save(dataset_dir / "targets.npy", np.concatenate(targets))
    np.save(dataset_dir / "c_indices.npy", np.concatenate(c_indices))
    if cthresholds:
        np.save(dataset_dir / "cthresholds.npy", np.array(cthresholds))
    np.save(dataset_dir / "event_row_splits.npy", row_splits)


# ======================================================================
def build_dataset(
    fn,
    batch_size,
    nev=-1,
    min_hits=1,
    split=None,
    is_training=False,
    cthreshold=None,
    should_save_dataset=False,
    should_load_dataset=False,
    dataset_dir=None,
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
        - should_load_dataset: bool, wether to to load the dataset from directory
                               specified by runcard
        - should_save_dataset: bool, wether to to save the dataset in directory
                               specified by runcard
        - dataset_dir: Path, dataset directory

    Returns
    -------
        - list, of Event instances
        - list, of inputs and targets couples:
            inputs is list of np.arrays of shape=(nb_cluster_pairs, nb_features);
            targets is a list of np.arrays of shape=(nb_cluster_pairs,)
    """
    events = None if is_training and should_load_dataset else load_events(fn, nev, min_hits)
    if should_load_dataset:
        print("[+] Loading dataset ...")
        dataset_tuple = load_dataset(dataset_dir)
    else:
        print("[+] Generating dataset ...")
        dataset_tuple = generate_dataset(
            events,
            min_hits=min_hits,
            cthreshold=cthreshold,
            should_save_dataset=should_save_dataset,
            dataset_dir=dataset_dir,
        )
    exit()
    return get_generator(events, *dataset_tuple, batch_size, is_training, split)


# ======================================================================
def build_dataset_train(setup, should_load_dataset=False, should_save_dataset=False):
    """
    Wrapper function to build dataset for training. Implements validation
    splitting according to config file.

    Parameters
    ----------
        - should_load_dataset: bool, wether to to load the dataset from directory
                               specified by runcard
        - should_save_dataset: bool, wether to to save the dataset in directory
                               specified by runcard

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
    dataset_dir = setup["train"]["dataset_dir"]
    return build_dataset(
        fn,
        batch_size,
        nev=nev,
        min_hits=min_hits,
        split=split,
        is_training=True,
        should_save_dataset=should_save_dataset,
        should_load_dataset=should_load_dataset,
        dataset_dir=dataset_dir,
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
    """
    Return a dummy dataset to build the model first when loading.

    Note
    ----
    Do NOT use this generator for the whole inference pipeline. The cluster
    indices are dummy float values, not integers.
    """
    B = 32
    inputs = [np.random.rand(B * 50, nb_feats) for i in range(2)]
    targets = [np.random.rand(B * 50) for i in range(2)]
    data = (None, [inputs, targets, targets])
    return EventDataset(data, batch_size=B, is_training=True, verbose=0)


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
