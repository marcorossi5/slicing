import numpy as np
import tensorflow as tf
from .read_data import load_events

class EventDataset(tf.keras.utils.Sequence):
    def __init__(
        self,
        inputs,
        shuffle=False,
        seed=12345,
    ):
        self.inputs, self.targets = inputs
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


# ======================================================================
def build_dataset_train(setup):
    fn = setup["train"]["fn"]
    nev = setup["train"]["nev"]
    min_hits = setup["train"]["min_hits"]
    split = setup["dataset"]["split"]

    events = load_events(fn, nev, min_hits)

    inputs = []
    targets = []

    for ev in events:
        for plane in ev.planes:
            norm = len(set(plane.ordered_cluster_idx))
            norm_clusters = plane.ordered_cluster_idx / norm
            inputs.append(np.concatenate([plane.point_cloud, [norm_clusters]], axis=0).T)
            targets.append(plane.ordered_mc_idx)
    

    inputs = np.array(inputs, dtype=object)
    targets = np.array(targets, dtype=object)
    assert len(inputs) == len(
        targets
    ), f"Length of inputs and targets must match, got {len(inputs)} and {len(targets)}"

    # split dataset
    nb_events = len(inputs)
    perm = np.random.permutation(nb_events)
    nb_split = int(split * nb_events)

    train_inp = inputs[perm[:nb_split]]
    train_trg = targets[perm[:nb_split]]

    val_inp = inputs[perm[nb_split:]]
    val_trg = targets[perm[nb_split:]]

    return EventDataset([train_inp, train_trg]), EventDataset([val_inp, val_trg])


# ======================================================================
def build_dataset_test(setup):
    fn = setup["test"]["fn"]
    nev = setup["test"]["nev"]
    min_hits = setup["test"]["min_hits"]

    events = load_events(fn, nev, min_hits)


def build_dataset(setup, is_training=None):
    if is_training:
        return build_dataset_train(setup)
    else:
        return build_dataset_test(setup)