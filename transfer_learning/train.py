import argparse
import numpy as np
import matplotlib.pyplot as plt
from time import time as tm
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau

from slicerl.models import build_actor_model
from slicerl.RandLANet import RandLANet

from slicerl.read_data import load_Events_from_file

DTYPE = tf.float32
eps = tf.constant(np.finfo(np.float64).eps, dtype=DTYPE)

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
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    #----------------------------------------------------------------------
    def __getitem__(self, idx):
        index = self.indexes[idx]
        return self.inputs[index], self.targets[index]

    #----------------------------------------------------------------------
    def __len__(self):
        return len(self.inputs)

#======================================================================
class F1score(tf.keras.metrics.Metric):
    def __init__(self, name='F1-score', **kwargs):
        super(F1score, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        pass

    def result(self):
        pass

    def reset_states(self):
        pass

#======================================================================
def dice_loss(y_true,y_pred):
    iy_true = 1-y_true
    iy_pred = 1-y_pred
    num1 = tf.math.reduce_sum((y_true*y_pred), -1) + eps
    den1 = tf.math.reduce_sum(y_true*y_true + y_pred*y_pred, -1) + eps
    num2 = tf.math.reduce_sum(iy_true*iy_pred, -1) + eps
    den2 = tf.math.reduce_sum(iy_true*iy_true + iy_pred*iy_pred, -1) + eps
    return 1 - tf.math.reduce_mean(num1/den1 + num2/den2)

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
def build_dataset(fn, nev=-1, min_hits=1, augment=False):
    if isinstance(fn, str):
        events  = load_Events_from_file(fn, nev, min_hits)
    elif isinstance(fn, list):
        events  = load_Events_from_files(fn, nev, min_hits)
    else:
        raise NotImplementedError, f"please provide string or list, not {type(fn)}"
    inputs  = []
    targets = []
    for event in events:
        index = -1
        while True:        
        # for index in range(10):
            state = event.state()
            if not event.nconsidered:
                break
            index += 1
            pc = state[:, 1:3]
            feats = state[:, ::3]
            # inputs.append(np.expand_dims(event.state(), 0))
            
            # generate cheating mask
            m = event.ordered_mc_idx[event.considered] == index
            
            # update status vector
            current_status = event.status[event.considered]
            current_status[m] = index
            event.status[event.considered] = current_status

            # pad and append to targets
            # padding = (0,event.max_hits-event.nconsidered)
            # targets.append( np.pad(m, padding) )

            # augment inputs with rotations
            if augment:
                pc, feats, target = transform(pc, feats, m)
            else:
                pc = pc[None]
                feats = feats[None]
                target = m[None]

            # print(f"pc shape: {pc.shape} \t feats shape: {feats.shape} \t targets shape:{target.shape}")
            inputs.append([pc, feats])
            targets.append(target)
    # return np.stack(inputs), np.stack(targets)
    return inputs, targets

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
def main():
    arger = argparse.ArgumentParser(
        """
    Example script to train RandLA-Net for slicing
    """
    )
    arger.add_argument(
        "-e", "--epochs", help="Number of events to be run", type=int, default=2
    )
    arger.add_argument(
        "-l", "--lr", help="Set the lerning rate", type=float, default=1e-2
    )
    arger.add_argument(
        "-n", "--nev", help="Number of events to be run", type=int, default=-1
    )
    arger.add_argument(
        "--output", help="Output folder", type=str, default='../transfer_learning/test'
    )
    args = arger.parse_args()

    folder = args.output
    # load train data
    fn       = [
        'data/test_data_05GeV.csv.gz',
        'data/test_data_1GeV.csv.gz',
        'data/test_data_2GeV.csv.gz',
        'data/test_data_3GeV.csv.gz',
        'data/test_data_6GeV.csv.gz',
        'data/test_data_7GeV.csv.gz',
        ]
    nev      = args.nev
    min_hits = 16
    train = build_dataset(fn, nev=nev, min_hits=min_hits)
    train_generator = EventDataset(train, shuffle=True)

    # load val, test data
    fn       = 'data/test_data_03GeV.csv.gz'
    nev      = 1
    min_hits = 16
    data = build_dataset(fn, nev=nev, min_hits=min_hits)
    (x_test, y_test), val = split_dataset(data)

    val_generator  = EventDataset(val, shuffle=False)
    test_generator = EventDataset((x_test, y_test), shuffle=False)

    # input_dim = (1,15000,4)
    batch_size = 1
    # actor = build_actor_model(None, input_dim)
    actor = RandLANet(nb_layers=4, activation=tf.nn.leaky_relu, name='RandLA-Net')

    lr     = args.lr
    epochs = args.epochs
    t      = 0.5
    actor.compile(
            loss= dice_loss,
            optimizer=tf.keras.optimizers.Adam(lr),
            metrics=[tf.keras.metrics.Precision(thresholds=t),
                     tf.keras.metrics.Recall(thresholds=t)]
            )
    
    actor.model().summary()
    # tf.keras.utils.plot_model(actor.model(), to_file='../RandLA-Net.png', expand_nested=True, show_shapes=True)

    logdir = f"{folder}/logs"
    callbacks = [
        TensorBoard(log_dir=logdir,
                    write_images=True,
                    profile_batch=2),
        ModelCheckpoint(f"{folder}"+"/actor.h5",
                        save_best_only=True,
                        mode='max',
                        monitor='val_precision',
                        verbose=1),
        ReduceLROnPlateau(monitor='val_precision', factor=0.5, mode='max',
                          verbose=1, patience=2, min_lr=1e-4)
        
    ]
    actor.fit(train_generator, epochs=epochs,
              validation_data=val_generator,
              callbacks=callbacks,
              verbose=2)

    
    results = actor.evaluate(test_generator)
    print("Test loss, test precision, test recall: ", results)

    y_pred = actor.get_prediction(x_test)
    # print(f"Start feats shape: {y_pred[0][0].shape} \t range: [{y_pred[0][0].min()}, {y_pred[0][0].max()}]")

    from slicerl.diagnostics import vnorm, vcmap

    pc      = x_test[0][0][0] # shape=(N,2)
    pc_pred = y_pred[0][0]    # shape=(N,)
    pc_test = y_test[0][0]    # shape=(N,)

    print(f"pc_pred shape: {pc_pred.shape} \t range: [{pc_pred.min()}, {pc_pred.max()}]")

    ax = plt.subplot(131)
    ax.scatter(pc[:,0], pc[:,1], s=0.5, c=pc_pred, cmap=vcmap, norm=vnorm)
    ax.set_title("pc_pred")

    ax = plt.subplot(132)
    ax.scatter(pc[:,0], pc[:,1], s=0.5, c=pc_pred>0.5, cmap=vcmap, norm=vnorm)
    ax.set_title("pc_pred > 0.5")

    ax = plt.subplot(133)
    ax.scatter(pc[:,0], pc[:,1], s=0.5, c=pc_test, cmap=vcmap, norm=vnorm)
    ax.set_title("pc_true")
    plt.savefig(f"{folder}/test.png", bbox_inches='tight', dpi=300)
    # plt.show()


if __name__=='__main__':
    start = tm()
    main()
    print(f"Program done in {tm()-start} s")