# This file is part of SliceRL by M. Rossi
from slicerl.config import NP_DTYPE
from slicerl.read_data import load_Events_from_file, load_Events_from_files
from slicerl.tools import onehot, onehot_to_indices
from slicerl.layers import LocSE

import tensorflow as tf
import numpy as np
import knn

#======================================================================
class EventDataset(tf.keras.utils.Sequence):
    """ Class defining dataset. """
    #----------------------------------------------------------------------
    def __init__(self, data, K, shuffle=True, **kwargs):
        """
        Parameters
        ----------
            - data    : list, [inputs, targets] for RandLA-Net
            - K       : int, neighbours number in KNN graph
            - shuffle : bool, wether to shuffle dataset on epoch end
        """
        # needed to generate the dataset
        self.__events = data[0]
        self.inputs, self.targets = data[1]
        self.K = K
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.inputs))
        assert len(self.inputs) == len(self.targets), \
            f"Length of inputs and targets must match, got {len(self.inputs)} and {len(self.targets)}"

        self.preprocess()

    #----------------------------------------------------------------------
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    #----------------------------------------------------------------------
    def __getitem__(self, idx):
        index = self.indexes[idx]
        return self.prep_inputs[index], self.prep_targets[index]

    #----------------------------------------------------------------------
    def __len__(self):
        return len(self.prep_inputs)

    #----------------------------------------------------------------------
    def preprocess(self):
        """
        Preprocess dataset for SEAC-Net.

        Returns
        -------
            - np.array, KNN indices of shape=(B,N,K)
        """
        prep_inputs   = []
        prep_targets  = []
        self.knn_idxs = []
        for (pc, feats), targets in zip(self.inputs, self.targets):
            # select knn neighbours
            knn_idx = knn.knn_batch(pc, pc, K=1+self.K,omp=True)
            self.knn_idxs.append(knn_idx)
            pc    = LocSE.gather_neighbours(pc, knn_idx).numpy()
            feats = LocSE.gather_neighbours(feats, knn_idx).numpy()
            prep_inputs.append([pc,feats])

            # produce targets
            tgt   = onehot_to_indices(targets)[...,None]
            tgt   = LocSE.gather_neighbours(tgt, knn_idx).numpy().squeeze(-1)
            mask  = (tgt[:,:,:1] == tgt).astype(NP_DTYPE)
            prep_targets.append(mask)

        self.prep_inputs  = prep_inputs
        self.prep_targets = prep_targets
        # print(self.prep_inputs[0][0].shape, self.prep_inputs[0][1].shape, self.prep_targets[0].shape)


    #----------------------------------------------------------------------
    def get_pc(self, index):
        """
        Get the point cloud at a certain index.

        Returns
        -------
            - np.array, point cloud of shape=(N,2)
        """
        return self.inputs[index][0][0]

    #----------------------------------------------------------------------
    def get_knn_idx(self, index, K):
        """
        Get the KNN graph indices for a pc at a certain index.

        Returns
        -------
            - np.array, KNN indices of shape=(N,K)
        """
        pc = self.inputs[index][0][0]
        return knn.knn_batch(pc, pc, K=K,omp=True)

    #----------------------------------------------------------------------
    def get_feats(self, index):
        """
        Get the point cloud features at a certain index.

        Returns
        -------
            - np.array, point cloud of shape=(N,2)
        """
        return self.inputs[index][1][0]

    #----------------------------------------------------------------------
    def get_onehot_targets(self, index):
        """
        Get the point cloud features at a certain index.

        Returns
        -------
            - np.array, point cloud of shape=(N, nb_classes)
        """
        return self.targets[index][0]

    #----------------------------------------------------------------------
    def get_targets(self, index):
        """
        Get the point cloud features at a certain index.

        Returns
        -------
            - np.array, point cloud of shape=(N,)
        """
        return onehot_to_indices( self.targets[index][0] )

    #----------------------------------------------------------------------
    @property
    def events(self):
        """ Event attribute. """
        return self.__events

    #----------------------------------------------------------------------
    @events.setter
    def events(self, y_pred):
        """
        Parameters
        ----------
            - y_pred: Predictions, object storing network predictions
        """
        print("[+] setting events")
        if self.__events:
            for i,event in enumerate(self.__events):
                event.store_preds(y_pred.get_pred(i))
        else:
            raise ValueError("Cannot set events attribute, found None"
                             " (is the EventDataset generator in training mode?)")


#======================================================================
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
    sint  = np.sin(t)
    cost  = np.cos(t)
    irots = np.array([[cost, sint],
                      [-sint, cost]])
    rots  = np.moveaxis(irots, [0,1,2], [2,1,0])
    return np.matmul(pc, rots)

#======================================================================
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
    fracs = np.array([0, 0.16,  0.25,  0.3,  0.5,  0.66,  0.75,  0.83, 1,
                        -0.16, -0.25, -0.3, -0.5, -0.66, -0.75, -0.83])
    t = np.pi*fracs
    # random permute the angles list to ensure no bias
    randt = np.random.permutation(t)
    B = len(t)

    # produce augmented pc
    pc = rotate_pc(pc, randt)

    # repeat feats and target along batch axis
    feats = np.repeat(feats[None], B, axis=0)
    target = np.repeat(target[None], B, axis=0)

    return pc, feats, target

#======================================================================
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
    assert len(inputs) == len(targets), \
            f"Length of inputs and targets must match, got {len(inputs)} and {len(targets)}"
    l = len(inputs)
    split_idx= int(l*split)

    val_inputs    = inputs[:split_idx]
    test_inputs   = inputs[split_idx:]

    val_targets   = targets[:split_idx]
    test_targets  = targets[split_idx:]

    return [val_inputs, val_targets],     \
           [test_inputs, test_targets]

#======================================================================
def build_dataset(fn, nev=-1, min_hits=1, split=None, augment=False, nb_classes=128):
    """
    Parameters
    ----------
        - fn         : list, of str events filenames
        - nev        : int, number of events to take from each file
        - min_hits   : int, minimum hits per slice for dataset inclusion
        - split      : float, split percentage between train and val in events
        - augment    : bool, wether to augment training set with rotations
        - nb_classes : int, number of classes to segment events in

    Returns
    -------
        - list, of Event instances
        - list, of inputs and targets couples:
            inputs is a list of couples [pc, features], pc of shape=(B,N,2) and
            features of shape=(B,N);
            targets is a list on np.arrays of shape=(B,N, nb_classes)
    """
    if isinstance(fn, str):
        events  = load_Events_from_file(fn, nev, min_hits)
    elif isinstance(fn, list) or isinstance(fn, tuple):
        events  = load_Events_from_files(fn, nev, min_hits)
    else:
        raise NotImplementedError(f"please provide string or list, not {type(fn)}")
    inputs  = []
    targets = []
    for event in events:
        state = event.state()
        pc = state[:, 1:3]
        feats = state[:, ::3]
        m = event.ordered_mc_idx
        if augment:
            pc, feats, target = transform(pc, feats, m)
        else:
            pc = pc[None]
            feats = feats[None]
            target = m[None]
        target = onehot(target, nb_classes)
        inputs.extend([[p[None], f[None]] for p,f in zip(pc,feats)])
        targets.extend(np.split(target, len(target), axis=0))

    if split:
        nb_events = len(inputs)
        perm = np.random.permutation(nb_events)
        nb_split = int(split*nb_events)

        train_evt = None
        train_inp = [ inputs[i] for i in perm[:nb_split] ]
        train_trg = [ targets[i] for i in perm[:nb_split] ]

        val_evt   = None
        val_inp   = [ inputs[i] for i in perm[nb_split:] ]
        val_trg   = [ targets[i] for i in perm[nb_split:] ]

        return ([train_evt, [train_inp, train_trg]],
                [val_evt, [val_inp, val_trg]]
               )
    return events, [inputs, targets]

#======================================================================
def build_dataset_train(setup):
    """
    Wrapper function to build dataset for training. Implements validation
    splitting according to config file.

    Returns
    -------
        - EventDataset, object for training
        - EventDataset, object for validation (None if split percentage is None)
    """
    fn         = setup['train']['fn']
    nev        = setup['train']['nev']
    min_hits   = setup['train']['min_hits']
    nb_classes = setup['model']['nb_classes']
    split      = setup['dataset']['split']
    augment    = False if setup['scan'] or (not setup['dataset']['augment']) else True
    K = setup['model']['K']

    train, val = build_dataset(
                    fn, nev=nev, min_hits=min_hits, nb_classes=nb_classes,
                    split=split, augment=augment
                              )
    train_generator = EventDataset(train, K, shuffle=True)
    val_generator   = EventDataset(val, K) if split else None
    return train_generator, val_generator

#======================================================================
def build_dataset_test(setup):
    """
    Wrapper function to build dataset for testing.

    Returns
    -------
        - EventDataset, object for testing
    """
    fn         = setup['test']['fn']
    nev        = setup['test']['nev']
    min_hits   = setup['test']['min_hits']
    nb_classes = setup['model']['nb_classes']
    K = setup['model']['K']

    data = build_dataset(fn, nev=nev, min_hits=min_hits, nb_classes=nb_classes)
    return EventDataset(data, K)

#======================================================================
def dummy_dataset(nb_classes, K):
    """ Return a dummy dataset to build the model first when loading. """
    B = 1
    N = 256
    inputs  = [np.random.rand(B,N,2), np.random.rand(B,N,2)]
    targets = np.random.rand(B,N,nb_classes)
    data = (None, [ [inputs], [targets] ] )
    return EventDataset(data, K)
