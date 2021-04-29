# This file is part of SliceRL by M. Rossi
import numpy as np
import matplotlib.pyplot as plt
from time import time as tm

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from slicerl.RandLANet import RandLANet
from tensorflow.keras.optimizers import (
    Adam,
    SGD,
    RMSprop,
    Adagrad
)

from slicerl.tools import onehot_to_indices
from slicerl.build_dataset import EventDataset, build_dataset, split_dataset
from slicerl.losses import get_loss


def build_and_train_model(setup):
    start = tm()
    # load train data
    fn       = setup['train']['fn']
    nev      = setup['train']['nev']
    min_hits = setup['train']['min_hits']
    train = build_dataset(fn, nev=nev, min_hits=min_hits, augment=True)
    train_generator = EventDataset(train, shuffle=True)

    # load val, test data
    fn       = setup['test']['fn']
    nev      = setup['test']['nev']
    min_hits = setup['test']['min_hits']
    data = build_dataset(fn, nev=nev, min_hits=min_hits)
    (x_test, y_test), val = split_dataset(data)

    val_generator  = EventDataset(val, shuffle=False)
    test_generator = EventDataset((x_test, y_test), shuffle=False)

    net = RandLANet(**setup['model'], name='RandLA-Net')

    lr = setup['train']['lr']
    if setup['train']['optimizer'] == 'Adam':
        opt = Adam(lr=lr)
    elif setup['train']['optimizer']  == 'SGD':
        opt = SGD(lr=lr)
    elif setup['train']['optimizer'] == 'RMSprop':
        opt = RMSprop(lr=lr)
    elif setup['train']['optimizer'] == 'Adagrad':
        opt = Adagrad(lr=lr)

    net.compile(
            loss= get_loss(setup['train']),
            optimizer= opt,
            metrics=[tf.keras.metrics.CategoricalAccuracy(name='acc')],
            run_eagerly=setup.get('debug')
            )
    
    net.model().summary()
    # tf.keras.utils.plot_model(net.model(), to_file=f"{setup['output']}/Network.png", expand_nested=True, show_shapes=True)

    logdir = f"{setup['output']}/logs"
    checkpoint_filepath = f"{setup['output']}"+"/randla.h5"
    callbacks = [
        TensorBoard(
            log_dir=logdir,
            write_graph=False,
            write_images=True,
            histogram_freq=setup['train']['hist_freq'],
            profile_batch=5
        ),
        ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_best_only=True,
            mode='max',
            monitor='val_acc',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_acc', fnet=0.5, mode='max',
            verbose=1,
            patience=setup['train']['patience'],
            min_lr=setup['train']['min_lr']
        )
    ]
    print(f"[+] Train for {setup['train']['epochs']} epochs ...")
    net.fit(train_generator, epochs=setup['train']['epochs'],
              validation_data=val_generator,
              callbacks=callbacks,
              verbose=2)

    print("[+] done with training, load best weights")
    net.load_weights(checkpoint_filepath)
    
    results = net.evaluate(test_generator)
    print(f"Test loss: {results[0]:.5f} \t test accuracy: {results[1]}")

    y_pred, y_probs = net.get_prediction(x_test)
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