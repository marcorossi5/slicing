# This file is part of SliceRL by M. Rossi
import tensorflow as tf

#======================================================================
class F1score(tf.keras.metrics.Metric):
    """ Implementation of F1 score. """
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
