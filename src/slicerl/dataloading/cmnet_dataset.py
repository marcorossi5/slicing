# This file is part of SliceRL by M. Rossi
import logging
from tqdm import tqdm
from math import ceil
import numpy as np
import tensorflow as tf
from slicerl import PACKAGE
from .read_data import load_events
from .Event import get_cluster_features

logger = logging.getLogger(PACKAGE)


class EventDataset(tf.keras.utils.Sequence):
    """Class defining dataset."""

    # ----------------------------------------------------------------------
    def __init__(
        self,
        data,
        batch_size,
        cthresholds=None,
        is_training=False,
        shuffle=False,
        seed=12345,
        should_split_by_data=True,
        should_balance=False,
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
            - seed: int, the seed of the random generator shuffling on epoch end
        """
        # needed to generate the dataset
        self.__events = data[0]
        self.nb_clusters_list = (
            [ev.nb_plane_clusters for ev in self.__events]
            if self.__events is not None
            else None
        )
        self.inputs_ = (
            np.concatenate(data[1][0], axis=0)
            if is_training and not should_split_by_data
            else data[1][0]
        )
        self.targets_ = (
            np.concatenate(data[1][1])
            if is_training and not should_split_by_data
            else data[1][1]
        )
        self.c_indices = (
            np.concatenate(data[1][2])
            if is_training and not should_split_by_data
            else data[1][2]
        )
        self.batch_size = batch_size
        self.cthresholds = cthresholds
        self.is_training = is_training
        self.shuffle = shuffle
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        self.indexes = np.arange(len(self.inputs_))
        assert len(self.inputs_) == len(
            self.targets_
        ), f"Length of inputs and targets must match, got {len(self.inputs_)} and {len(self.targets_)}"
        if should_balance:
            nb_positive = np.count_nonzero(self.targets_)
            nb_all = len(self.targets_)
            balancing = nb_positive / nb_all
            logger.debug(f"Training points: {nb_all} of which positives: {nb_positive}")
            logger.debug(f"Percentage of positives: {balancing}")
            self.inputs, self.targets = balance_dataset(
                self.inputs_, self.targets_, balancing
            )
        else:
            self.inputs = self.inputs_
            self.targets = self.targets_
            # no balance needed for cluster indices because they enter inference only

        self.perm = np.arange(self.__len__())

    # ----------------------------------------------------------------------
    def on_epoch_end(self):
        if self.shuffle:
            self.perm = self.rng.permutation(self.__len__())

    # ----------------------------------------------------------------------
    def __getitem__(self, idx):
        ii = self.perm[idx]
        batch_x = self.inputs[ii * self.batch_size : (ii + 1) * self.batch_size]
        batch_y = self.targets[ii * self.batch_size : (ii + 1) * self.batch_size]
        return batch_x, batch_y

    # ----------------------------------------------------------------------
    def __len__(self):
        return ceil(len(self.inputs) / self.batch_size)

    # ----------------------------------------------------------------------
    @property
    def events(self):
        """Event attribute."""
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
                event.store_preds(y_pred.get_slices(i))
        else:
            raise ValueError(
                "Cannot set events attribute, found None"
                " (is the EventDataset generator in training mode?)"
            )


# ======================================================================
class MHAEventDataset(EventDataset):
    def __init__(self, *args, **kwargs):
        super(MHAEventDataset, self).__init__(*args, **kwargs)

    # ----------------------------------------------------------------------
    def __getitem__(self, idx):
        ii = self.perm[idx]
        batch_x = self.inputs[ii][None]
        batch_y = self.targets[ii : ii + 1]
        return batch_x, batch_y


# ======================================================================
class MHAEventDatasetFromNp(tf.keras.utils.Sequence):
    """Dataset from numpy array used for training only."""

    # ----------------------------------------------------------------------
    def __init__(self, inputs, targets, batch_size, shuffle=True, seed=12345):
        """
        Parameters
        ----------
            - inputs: np.array, inputs array each of shape=(nb clusters pairs, nb feats)
            - targets: np.array, target array of shape=(nb clusters pairs,)
            - batch_size: int, batch size
            - shuffle: bool, wether to shuffle inputs on epoch end
            - seed: int, the seed of the random generator shuffling on epoch end
        """
        self.inputs = inputs
        self.targets = targets
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.rng = np.random.default_rng(self.seed)
        self.perm = np.arange(self.__len__())

    # ----------------------------------------------------------------------
    def on_epoch_end(self):
        if self.shuffle:
            self.perm = self.rng.permutation(self.__len__())

    # ----------------------------------------------------------------------
    def __getitem__(self, idx):
        """
        Returns
        -------
            - np.array: batch of inputs of shape=(1, [nb hits], nb feats). The
                        feature axis contains for each hit:
                            - deposited energy
                            - x position
                            - z position
                            - expected x direction
                            - expected z direction
                            - cluster origin: 1 or 0
            - np.array: target label of shape (1,)
        """
        ii = self.perm[idx]
        batch_x = self.inputs[ii][None]
        batch_y = self.targets[ii : ii + 1]
        return batch_x, batch_y

    # ----------------------------------------------------------------------
    def __len__(self):
        return ceil(len(self.inputs) / self.batch_size)


# ======================================================================
def balance_dataset(inputs, targets):
    """
    Balances the dataset discarding part of the negative samples to match
    the size of the positive ones. This happens if the ratio of positve
    samples is below 20%.

    Parameters
    ----------
        - inputs: np.array, of shape=(nb_data, nb_feats)
        - targets: np.array, of shape=(nb_data,)
        - balancing: float, percentage of positive samples over the total
    """
    m_positives = targets.astype(bool)
    neg_idx = np.argwhere(~m_positives).flatten()
    neg_selected = np.random.choice(
        neg_idx, size=np.count_nonzero(m_positives), replace=False
    )
    m_negatives = np.isin(np.arange(len(targets)), neg_selected)
    mask = np.logical_or(m_positives, m_negatives)
    bal_inputs = inputs[mask]
    bal_targets = targets[mask]

    nb_pre_positive = np.count_nonzero(m_positives)
    nb_pre_all = len(targets)
    pre_balancing = nb_pre_positive / nb_pre_all
    nb_positive = np.count_nonzero(bal_targets)
    nb_all = len(bal_targets)
    balancing = nb_positive / nb_all
    logger.debug(
        "\n\tBefore balancing:\n"
        f"\tTraining points: {nb_pre_all} of which positives: {nb_pre_positive}\n"
        f"\nPercentage of positives: {pre_balancing}\n"
        "\nAfter balancing\n"
        f"\nTraining points: {nb_all} of which positives: {nb_positive}\n"
        f"\nPercentage of positives: {balancing}"
    )

    return bal_inputs, bal_targets


# ======================================================================
def split_dataset_by_data(data, split=0.5):
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
    inputs, targets, _ = data
    assert len(inputs) == len(
        targets
    ), f"Length of inputs and targets must match, got {len(inputs)} and {len(targets)}"
    inputs = np.concatenate(inputs, axis=0)
    targets = np.concatenate(targets, axis=0)

    inputs, targets = balance_dataset(inputs, targets)

    nb_events = len(inputs)
    perm = np.random.permutation(nb_events)
    nb_split = int(split * nb_events)

    train_inp = inputs[perm][:nb_split]
    train_trg = targets[perm][:nb_split]

    val_inp = inputs[perm][nb_split:]
    val_trg = targets[perm][nb_split:]

    return [train_inp, train_trg, None], [val_inp, val_trg, None]


# ======================================================================
def split_dataset_by_event(data, split=0.5):
    """
    Parameters
    ----------
        - data  : list, [inputs, targets, c indices] to split
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
def generate_inputs_and_targets_mha(event, min_hits=1):
    """
    Takes an Event object and processes the 2D clusters to retrieve inputs and
    associated target arrays. Input arrays are 2D clusters point clouds
    vectors. Target arrays are binaries. If n is cluster number in an event,
    n*(n-1)/2 clusters pairs are produced. Filters the clusters by number of hits.

    Parameters
    ----------
        - event: Event, the event object holding the hits information
        - is_training: bool

    Returns
    -------
        - np.array, of shape=(nb_clusters_pairs,). Each element is an np.array
                    of shape=(1, nb_hits, 6)
        - np.array, target labels of shape=(nb_clusters_pairs,)
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
                ilen = icluster.shape[1]

                jcluster = plane.state(j)
                jlen = jcluster.shape[1]

                if ilen < min_hits or jlen < min_hits:
                    # TODO: is this needed?
                    continue

                # add cluster origin feature
                ipc = np.concatenate([icluster, np.zeros([1, ilen])], axis=0)
                jpc = np.concatenate([jcluster, np.ones([1, jlen])], axis=0)

                # inputs of shape (1, nb hits, 6)
                inputs.append(np.concatenate([ipc, jpc], axis=1).T)

                # targets
                ipfo = plane.cluster_to_main_pfo[i]
                jpfo = plane.cluster_to_main_pfo[j]
                tgt = 1.0 if ipfo == jpfo else 0.0
                targets.append(tgt)

                # indices in the adj matrix
                c_indices.append([ped + i, ped + j])
        ped += plane.nb_clusters
    return np.array(inputs, dtype=object), np.array(targets), np.array(c_indices)


# ======================================================================
def get_generator(
    events,
    inputs,
    targets,
    c_indices,
    cthresholds,
    batch_size,
    is_training,
    split_by,
    split,
):
    """
    Get EventDataset generator from data.
    """
    kwargs = {
        "batch_size": batch_size,
        "is_training": is_training,
        "shuffle": False,
    }
    if split:
        assert split_by in [
            None,
            "data",
            "event",
        ], f"{split_by} option not recognized.Valid values are 'data' or 'event'"
        split_wrapper = (
            split_dataset_by_data if split_by == "data" else split_dataset_by_event
        )
        train_splitted, val_splitted = split_wrapper(
            [inputs, targets, c_indices], split
        )
        train_data = [None, train_splitted]
        val_data = [None, val_splitted]
        keys = kwargs.copy()
        keys.update(shuffle=True)
        return MHAEventDataset(train_data, **keys), MHAEventDataset(val_data, **kwargs)
    data = events, [inputs, targets, c_indices]
    return MHAEventDataset(data, cthresholds=cthresholds, **kwargs)


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
    for event in tqdm(events):  # tqdm(events):
        event.refine()
        inps, tgts, c_idxs = generate_inputs_and_targets_mha(event, min_hits=min_hits)
        if cthreshold is not None:
            cthresholds.append(get_cluster_thresholds(event, cthreshold))
        inputs.append(inps)
        targets.append(tgts)
        c_indices.append(c_idxs)
    if should_save_dataset:
        save_dataset(dataset_dir, inputs, targets, c_indices, cthresholds)
    return inputs, targets, c_indices, cthresholds


# ======================================================================
def _build_dataset(
    fn,
    batch_size,
    nev=-1,
    min_hits=1,
    split_by=None,
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
        - split_by: str, split by data or split by event. Default: None
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
    if not is_training or not should_load_dataset:
        events = load_events(fn, nev, min_hits)
        for event in events:
            event.refine(skip_feats_computation=True)
    else:
        events = None

    if should_load_dataset:
        logger.info("Loading dataset ...")
        dataset_tuple = load_dataset(dataset_dir)
    else:
        logger.info("Generating dataset ...")
        dataset_tuple = generate_dataset(
            events,
            min_hits=min_hits,
            cthreshold=cthreshold,
            should_save_dataset=should_save_dataset,
            dataset_dir=dataset_dir,
        )
    return get_generator(
        events, *dataset_tuple, batch_size, is_training, split_by, split
    )


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
    split_by = setup["train"]["split_by"]
    return _build_dataset(
        fn,
        batch_size,
        nev=nev,
        min_hits=min_hits,
        split=split,
        is_training=True,
        split_by=split_by,
        should_save_dataset=should_save_dataset,
        should_load_dataset=should_load_dataset,
        dataset_dir=dataset_dir,
    )


# ======================================================================
def build_dataset_test(setup, should_save_dataset=False, should_load_dataset=False):
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
    dataset_dir = setup["test"]["dataset_dir"]

    return _build_dataset(
        fn,
        batch_size,
        nev=nev,
        min_hits=min_hits,
        cthreshold=cthreshold,
        should_save_dataset=should_save_dataset,
        should_load_dataset=should_load_dataset,
        dataset_dir=dataset_dir,
    )


# ======================================================================
def build_dataset_from_np(setup, folder):
    """
    Loads generators from saved numpy arrays.

    Parameters
    ----------
        - folder: Path, the folder containing training numpy arrays

    Returns
    -------
        - tuple, (train_generator, val_generator)

    """
    batch_size = setup["model"]["batch_size"]
    min_hits = setup["train"]["min_hits"]
    train_inputs = np.load(folder / "train_inputs.npy", allow_pickle=True)
    train_targets = np.load(folder / "train_targets.npy")
    val_inputs = np.load(folder / "val_inputs.npy", allow_pickle=True)
    val_targets = np.load(folder / "val_targets.npy")

    fn = np.vectorize(lambda x: x.shape[0])
    m = fn(train_inputs) > 2 * min_hits

    return MHAEventDatasetFromNp(
        train_inputs[m], train_targets[m], batch_size
    ), MHAEventDatasetFromNp(val_inputs, val_targets, batch_size)


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
    cthresholds = np.load(fname).astype(int) if fname.is_file() else []

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
def save_dataset_np(generators, folder):
    """
    Parameters
    ----------
        - generators: tuple, (train, val) generators of MHAEventDataset objects
        - folder: Path
    """
    train, val = generators
    train_inputs = np.array(train.inputs, dtype=object)
    val_inputs = np.array(val.inputs, dtype=object)
    np.save(folder / "train_inputs.npy", train_inputs)
    np.save(folder / "train_targets.npy", train.targets)
    np.save(folder / "val_inputs.npy", val_inputs)
    np.save(folder / "val_targets.npy", val.targets)


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
