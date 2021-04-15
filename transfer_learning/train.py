import numpy as np
import matplotlib.pyplot as plt
from time import time as tm
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau

from slicerl.models import build_actor_model
from slicerl.read_data import load_Events_from_file

DTYPE = tf.float32
eps = tf.constant(np.finfo(np.float64).eps, dtype=DTYPE)

class F1score(tf.keras.metrics.Metric):
    def __init__(self, name='F1-score', **kwargs):
        super(F1score, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        pass

    def result(self):
        pass

    def reset_states(self):
        pass

def dice_loss(y_true,y_pred):
    iy_true = 1-y_true
    iy_pred = 1-y_pred
    num1 = tf.math.reduce_sum((y_true*y_pred), -1) + eps
    den1 = tf.math.reduce_sum(y_true*y_true + y_pred*y_pred, -1) + eps
    num2 = tf.math.reduce_sum(iy_true*iy_pred, -1) + eps
    den2 = tf.math.reduce_sum(iy_true*iy_true + iy_pred*iy_pred, -1) + eps
    return 1 - tf.math.reduce_mean(num1/den1 + num2/den2)


def generate_dataset(fn, nev=-1, min_hits=1):
    events  = load_Events_from_file(fn, nev, min_hits)
    inputs  = []
    targets = []
    for event in events:
        for index in range(10):
            inputs.append(np.expand_dims(event.state(), 0))
            
            # generate cheating mask
            m = event.ordered_mc_idx[event.considered] == index
            
            # update status vector
            current_status = event.status[event.considered]
            current_status[m] = index
            event.status[event.considered] = current_status

            # pad and append to targets
            padding = (0,event.max_hits-event.nconsidered)
            targets.append( np.pad(m, padding) )
    return np.stack(inputs), np.stack(targets)

def main():
    folder = '../transfer_learning/test'
    # load data
    fn       = 'test_data_2GeV.csv.gz'
    nev      = -1
    min_hits = 300
    inputs, targets = generate_dataset(fn, nev=nev, min_hits=min_hits)

    # fn = 'test_data_03GeV.csv.gz'
    # inputs0, targets0 = generate_dataset(fn)

    # inputs = np.concatenate([inputs, inputs0])
    # del inputs0
    # targets = np.concatenate([targets, targets0])
    # del targets0

    assert len(inputs) == len(targets)

    train_p = 0.8 # train percentage
    val_p   = 0.1 # validation percentage
    test_p  = 0.1 # test percentage

    # train_p = 0.9 # train percentage
    # val_p   = 0.05 # validation percentage
    # test_p  = 0.05 # test percentage
    assert train_p + val_p + test_p == 1.

    splitting = [int(inputs.shape[0]*train_p), int(inputs.shape[0]*(train_p+val_p))]
    x_train, x_val, x_test = np.split(inputs, splitting)
    y_train, y_val, y_test = np.split(targets, splitting)

    input_dim = (1,15000,4)
    batch_size = 8
    actor = build_actor_model(None, input_dim)

    lr = 1e-3
    epochs = 50
    actor.compile(
            loss= dice_loss,
            optimizer=tf.keras.optimizers.Adam(lr),
            metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            )

    logdir = f"{folder}/logs"
    callbacks = [
        TensorBoard(log_dir=logdir),
        ModelCheckpoint(f"{folder}"+"/actor.h5",
                        save_best_only=True,
                        mode='max',
                        monitor='val_recall',
                        verbose=1),
        ReduceLROnPlateau(monitor='val_recall', factor=0.5, mode='max',
                          verbose=1, patience=5, min_lr=1e-4)
        
    ]
    actor.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_data=(x_val, y_val),
              callbacks=callbacks,
              verbose=2)
    
    results = actor.evaluate(x_test, y_test, batch_size=128)
    print("Test loss, test precision, test recall: ", results)

    y_pred = actor.predict(x_test)

    plt.subplot(131)
    m = x_test[0,0,:,1] > -1
    plt.scatter(x_test[0,0,:,1][m], x_test[0,0,:,2][m], s=0.5, c=y_pred[0][m])

    plt.subplot(132)
    plt.scatter(x_test[0,0,:,1][m], x_test[0,0,:,2][m], s=0.5, c=y_pred[0][m]>0.5)

    plt.subplot(133)
    plt.scatter(x_test[0,0,:,1][m], x_test[0,0,:,2][m], s=0.5, c=y_test[0][m])
    plt.show()


if __name__=='__main__':
    start = tm()
    main()
    print(f"Program done in {tm()-start} s")