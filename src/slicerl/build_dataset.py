# This file is part of SliceRL by M. Rossi
import tensorflow as tf
import numpy as np

from slicerl.read_data import load_Events_from_file, load_Events_from_files
from slicerl.tools import onehot, onehot_to_indices

#======================================================================
class EventDataset(tf.keras.utils.Sequence):
    """ Class defining dataset. """
    #----------------------------------------------------------------------
    def __init__(self, data, shuffle=True, **kwargs):
        """
        Parameters
        ----------
            - data    : list, [inputs, targets] for RandLA-Net
            - shuffle : bool, wether to shuffle dataset on epoch end
        """
        # needed to generate the dataset
        self.inputs, self.targets = data
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.inputs))
        assert len(self.inputs) == len(self.targets), \
            f"Length of inputs and targets must match, got {len(self.inputs)} and {len(self.targets)}"
    
    #----------------------------------------------------------------------
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    #----------------------------------------------------------------------
    def __getitem__(self, idx):
        index = self.indexes[idx]
        return self.inputs[index], self.targets[index]

    #----------------------------------------------------------------------
    def __len__(self):
        return len(self.inputs)
    
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
    sint = np.sin(t)
    cost = np.cos(t)
    irots = np.array([[cost, sint],
                      [-sint, cost]])
    rots = np.moveaxis(irots, [0,1,2], [2,1,0])
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
def build_dataset(fn, nev=-1, min_hits=1, augment=False, nb_classes=128):
    """
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
    elif isinstance(fn, list):
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
    return events, [inputs, targets]

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
