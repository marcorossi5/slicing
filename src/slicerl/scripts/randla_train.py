import os, argparse, shutil
import numpy as np
import matplotlib.pyplot as plt
from time import time as tm
import tensorflow as tf

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from slicerl.RandLANet import RandLANet

from slicerl.read_data import load_Events_from_file, load_Events_from_files

from slicerl.tools import onehot, onehot_to_indices

DTYPE = tf.float32
eps = tf.constant(np.finfo(np.float64).eps, dtype=DTYPE)

#----------------------------------------------------------------------
def makedir(folder):
    """Create directory."""
    if not os.path.exists(folder):
        os.mkdir(folder)
    else:
        raise Exception('Output folder %s already exists.' % folder)

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

#======================================================================
class F1score(tf.keras.metrics.Metric):
    def __init__(self, name='F1-score', **kwargs):
        super(F1score, self).__init__(name=name, **kwargs)

    #----------------------------------------------------------------------
    def update_state(self, y_true, y_pred, sample_weight=None):
        pass

    #----------------------------------------------------------------------
    def result(self):
        pass

    #----------------------------------------------------------------------
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
def build_dataset(fn, nev=-1, min_hits=1, augment=False, nb_classes=128):
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
    parser = argparse.ArgumentParser(
        """
    Example script to train RandLA-Net for slicing
    """
    )
    parser.add_argument(
        "--output", help="Output folder", type=str, default=None
    )
    parser.add_argument(
        '--force', '-f', help='Overwrite existing files if present', action='store_true', dest='force',                        
    )
    parser.add_argument(
        "-e", "--epochs", help="Number of events to be run", type=int, default=2
    )
    parser.add_argument(
        "-l", "--lr", help="Set the lerning rate", type=float, default=1e-2
    )
    parser.add_argument(
        "-n", "--nev", help="Number of events to be run", type=int, default=-1
    )
    parser.add_argument(
        "--nb_layers", help="Number of RandLA-Net layers", type=int, default=4
    )
    parser.add_argument(
        "--debug", help="Run TensorFlow eagerly", action="store_true", default=False
    )
    args = parser.parse_args()

    setup = {}
    

    if args.debug:
        print("[+] Run all tf functions eagerly")
        tf.config.run_functions_eagerly(True)

    # create output folder
    if args.output is not None:
        out = args.output
    try:
        makedir(out)
    except Exception as error:
        if args.force:
            print(f'WARNING: Overwriting {out} with new model')
            shutil.rmtree(out)
            makedir(out)
        else:
            print(error)
            print('Delete or run with "--force" to overwrite.')
            exit(-1)

    setup['output'] = out
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
    train = build_dataset(fn, nev=nev, min_hits=min_hits, augment=True)
    train_generator = EventDataset(train, shuffle=True)

    # load val, test data
    fn       = 'data/test_data_03GeV.csv.gz'
    nev      = -1
    min_hits = 16
    data = build_dataset(fn, nev=nev, min_hits=min_hits)
    (x_test, y_test), val = split_dataset(data)

    val_generator  = EventDataset(val, shuffle=False)
    test_generator = EventDataset((x_test, y_test), shuffle=False)

    batch_size = 1
    actor = RandLANet(nb_layers=args.nb_layers, activation=tf.nn.leaky_relu, fc_type='conv', name='RandLA-Net')

    t      = 0.5
    actor.compile(
            loss= tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(args.lr),
            metrics=[tf.keras.metrics.CategoricalAccuracy(name='acc')],
            run_eagerly=args.debug
            )
    
    actor.model().summary()
    # tf.keras.utils.plot_model(actor.model(), to_file=f"{setup['output']}/Network.png", expand_nested=True, show_shapes=True)

    logdir = f"{setup['output']}/logs"
    checkpoint_filepath = f"{setup['output']}"+"/actor.h5"
    callbacks = [
        TensorBoard(log_dir=logdir,
                    write_graph=False,
                    write_images=True,
                    histogram_freq=5,
                    profile_batch=5),
        ModelCheckpoint(filepath=checkpoint_filepath,
                        save_best_only=True,
                        mode='max',
                        monitor='val_acc',
                        verbose=1),
        ReduceLROnPlateau(monitor='val_acc', factor=0.5, mode='max',
                          verbose=1, patience=5, min_lr=1e-4)
    ]
    print(f"[+] Train for {args.epochs} epochs ...")
    actor.fit(train_generator, epochs=args.epochs,
              validation_data=val_generator,
              callbacks=callbacks,
              verbose=2)

    print("[+] done with training, load best weights")
    actor.load_weights(checkpoint_filepath)
    
    results = actor.evaluate(test_generator)
    print(f"Test loss: {results[0]:.5f} \t test accuracy: {results[1]}")

    y_pred, y_probs = actor.get_prediction(x_test)
    # print(f"Feats shape: {y_pred[0][0].shape} \t range: [{y_pred[0][0].min()}, {y_pred[0][0].max()}]")

    from slicerl.diagnostics import norm, cmap

    pc      = x_test[0][0][0]                    # shape=(N,2)
    pc_pred = y_pred[0][0]                       # shape=(N,)
    pc_test = onehot_to_indices(y_test[0][0])    # shape=(N,)
    # print(f"pc shape: {pc.shape} \t pc pred shape: {pc_pred.shape} \t pc test shape: {pc_test.shape}")

    fig = plt.figure(figsize=(18*2,14))
    ax = fig.add_subplot(121)
    ax.scatter(pc[:,0], pc[:,1], s=0.5, c=pc_pred, cmap=cmap, norm=norm)
    ax.set_title("pc_pred")

    ax = fig.add_subplot(122)
    ax.scatter(pc[:,0], pc[:,1], s=0.5, c=pc_test, cmap=cmap, norm=norm)
    ax.set_title("pc_true")
    fname = f"{setup['output']}/test.png"
    plt.savefig(fname, bbox_inches='tight', dpi=300)
    print(f"[+] Saved plot at {fname}")
    # plt.show()


if __name__=='__main__':
    start = tm()
    main()
    print(f"Program done in {tm()-start} s")

# TODO: add the possibility to weight the cross entropy to give more importance
# bigger slices than smaller ones