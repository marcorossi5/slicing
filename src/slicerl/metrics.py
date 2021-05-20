# This file is part of SliceRL by M. Rossi
from slicerl.config import float_me
from slicerl.tools import onehot

import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import(
    Metric,
    CategoricalCrossentropy,
    CategoricalAccuracy
)

#======================================================================
class F1score(Metric):
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

#======================================================================
class WCCEMetric(CategoricalCrossentropy):
    """ Wrapper class for weighted categorical crossentropy. """
    def __init__(self, nb_classes, scale, name='w_xent', **kwargs):
        super(WCCEMetric, self).__init__(name=name, **kwargs)
        self.nb_classes = nb_classes
        self.scale      = scale
        x = np.linspace(0, self.nb_classes-1, self.nb_classes)
        wgt = np.exp(- x/self.scale).reshape([1,1,-1])
        self.wgt = float_me(wgt)
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        weighted_true = y_true * self.wgt
        return super().update_state(weighted_true, y_pred, sample_weight)

#======================================================================
class WAccMetric(Metric):
    """ Wrapper class for weighted categorical accuracy. """
    def __init__(self, nb_classes, scale, name='w_acc', **kwargs):
        super(WAccMetric, self).__init__(name=name, **kwargs)
        self.nb_classes = nb_classes
        self.scale      = scale
        self.fzero = tf.zeros(shape=(self.nb_classes,))
        x = np.linspace(0, self.nb_classes-1, self.nb_classes)
        wgt = np.exp(- x/self.scale).reshape([1,1,-1])
        normed_wgt = wgt / wgt.sum()
        self.wgt = float_me(normed_wgt)

        # define the good predictions
        self.hits = self.add_weight(
                            name="hits",
                            initializer="zeros"
                                   )

        # store the probability normalization
        self.normalization = self.add_weight(
                                    name="normalization",
                                    initializer="zeros"
                                            )
    
    def update_state(self, y_true, y_pred, sample_weight):
        y_pred_max = tf.argmax(y_pred, axis=-1)
        y_true_max = tf.argmax(y_true, axis=-1)
        mask = tf.cast(y_true_max, "int32") == tf.cast(y_pred_max, "int32")

        y_true = tf.reduce_sum(self.wgt * y_true, axis=-1)
        values = tf.boolean_mask(y_true, mask)
        values = tf.cast(values, "float32")

        if sample_weight is not None:
            pass
        
        self.hits.assign_add( tf.reduce_sum(values) )        
        self.normalization.assign_add( tf.reduce_sum(y_true) )
    
    def result(self):
        return tf.reduce_sum(self.hits) / tf.reduce_sum(self.normalization)
    
    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.hits.assign(0.0)
        self.normalization.assign(0.0)
