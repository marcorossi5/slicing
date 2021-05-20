# This file is part of SliceRL by M. Rossi
from slicerl.RandLANet import RandLANet
from slicerl.DRBNet import DRBNet
from slicerl.build_dataset import dummy_dataset
from slicerl.losses import get_loss
from slicerl.metrics import WAccMetric
from slicerl.diagnostics import (
    plot_plane_view,
    plot_slice_size,
    plot_multiplicity
)

import os
from time import time as tm

import tensorflow as tf
from tensorflow.keras.callbacks import (
    TensorBoard,
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping
)
from tensorflow.keras.optimizers import (
    Adam,
    SGD,
    RMSprop,
    Adagrad
)

from hyperopt import STATUS_OK

def load_network(setup, checkpoint_filepath=None):
    """
    Load network from config dic, compile and, if necessary, load weights.

    Parameters
    ----------
        - setup               : dict, config dict
        - checkpoint_filepath : str, model checkpoint path

    Returns
    -------
        - RandLANet or DRBNet
    """
    net_dict = {}
    # update common keys
    net_keys = ['nb_classes', 'K', 'use_bias', 'fc_type', 'dropout', 'use_ggf']
    for key in net_keys:
        net_dict[key] = setup['model'][key]

    if setup['model']['net_type'] == 'RandLA':
        # update local keys
        net_dict['scale_factor'] = setup['model']['scale_factor']
        net_dict['nb_layers'] = setup['model']['nb_layers']
        net = RandLANet(name='RandLA-Net', **net_dict)
    elif setup['model']['net_type'] == 'DRB':
        # update local keys
        net_dict['nb_final_convs'] = setup['model']['nb_final_convs']
        net = DRBNet(name='DRB-Net', **net_dict)

    loss = get_loss(setup['train'], setup['model']['nb_classes'])

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
            loss= loss,
            optimizer= opt,
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(name='acc'),
                WAccMetric(
                    setup['model']['nb_classes'], setup['train']['wgt'], name='w_acc'
                          )
            ],
            run_eagerly=setup.get('debug')
            )

    net.model().summary()
    # tf.keras.utils.plot_model(net.model(), to_file=f"{setup['output']}/Network.png", expand_nested=True, show_shapes=True)

    if checkpoint_filepath:
        print(f"[+] Loading weights at {checkpoint_filepath}")
        dummy_generator = dummy_dataset(setup['model']['nb_classes'])
        net.evaluate(dummy_generator, verbose=0)
        net.load_weights(checkpoint_filepath)

    return net

#======================================================================
def build_and_train_model(setup, generators):
    """
    Train a model. If setup['scan'] is True, then perform hyperparameter search.

    Parameters
    ----------
        - setup      : dict
        - generators : list, of EventDataset with train and val generators

    Retruns
    -------
        RandLANet model if scan is False, else dict with loss and status keys.
    """
    train_generator, val_generator = generators

    net = load_network(setup)

    logdir = setup['output'].joinpath(f'logs/{tm()}').as_posix()
    checkpoint_filepath = setup['output'].joinpath(f'randla.h5').as_posix()
    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_best_only=True,
            mode='max',
            monitor='val_w_acc',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_acc', factor=0.5, mode='max',
            verbose=1,
            patience=setup['train']['patience'],
            min_lr=setup['train']['min_lr']
        ),
        EarlyStopping(
            monitor='val_acc',
            min_delta=0.001,
            mode='max',
            patience=15,
            restore_best_weights=True
        )
    ]
    if setup['scan']:
        tboard = TensorBoard(log_dir=logdir, profile_batch=0)
    else:
        tboard = TensorBoard(
                    log_dir=logdir,
                    write_graph=False,
                    write_images=True,
                    histogram_freq=setup['train']['hist_freq'],
                    profile_batch=5
                 )

    callbacks.append(tboard)

    initial_weights = setup['train']['initial_weights']
    if initial_weights and os.path.isfile(initial_weights):
        # dummy forward pass
        net.get_prediction(test_generator.inputs)
        net.load_weights(setup['train']['initial_weights'])
        print(f"[+] Found Initial weights configuration at {initial_weights} ... ")

    print(f"[+] Train for {setup['train']['epochs']} epochs ...")
    r = net.fit(train_generator, epochs=setup['train']['epochs'],
                validation_data=val_generator,
                callbacks=callbacks,
                verbose=2)

    if setup['scan']:
        net.load_weights(checkpoint_filepath)
        loss, acc = net.evaluate(val_generator, verbose=0)
        print(f"Evaluate model instance: [loss: {loss:.5f}, acc: {acc:.5f}]")
        res = {'loss': loss, 'acc': acc, 'status': STATUS_OK}
    else:
        res = net
    return res

#----------------------------------------------------------------------

def inference(setup, test_generator):
    print("[+] done with training, load best weights")
    checkpoint_filepath = setup['output'].joinpath('randla.h5')
    net = load_network(setup, checkpoint_filepath.as_posix())

    results = net.evaluate(test_generator)
    print(f"Test loss: {results[0]:.5f} \t test accuracy: {results[1]}")

    y_pred = net.get_prediction(test_generator.inputs)
    # print(f"Feats shape: {y_pred[0][0].shape} \t range: [{y_pred[0][0].min()}, {y_pred[0][0].max()}]")
    test_generator.events = y_pred

    plot_slice_size(test_generator.events, setup['output'].joinpath('plots'))
    plot_multiplicity(test_generator.events, setup['output'].joinpath('plots'))

    n = min(10, len(test_generator))
    for i in range(n):
        pc      = test_generator.get_pc(i)      # shape=(N,2)
        pc_pred = y_pred.get_pred(i)            # shape=(N,)
        pc_test = test_generator.get_targets(i) # shape=(N,)
        plot_plane_view(pc, pc_pred, pc_test, i, setup['output'].joinpath('plots'))

    # print(f"pc shape: {pc.shape} \t pc pred shape: {pc_pred.shape} \t pc test shape: {pc_test.shape}")
