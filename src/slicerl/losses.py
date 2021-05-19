# This file is part of SliceRL by M. Rossi
from slicerl.config import EPS, float_me

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import (
    Loss,
    CategoricalCrossentropy,
    MAE,
    Reduction
)

eps = float_me(EPS)

#======================================================================
def dice_loss(y_true,y_pred):
    """ Implementation of Dice loss. """
    iy_true = 1-y_true
    iy_pred = 1-y_pred
    num1 = tf.math.reduce_sum((y_true*y_pred), -1) + eps
    den1 = tf.math.reduce_sum(y_true*y_true + y_pred*y_pred, -1) + eps
    num2 = tf.math.reduce_sum(iy_true*iy_pred, -1) + eps
    den2 = tf.math.reduce_sum(iy_true*iy_true + iy_pred*iy_pred, -1) + eps
    return 1 - tf.math.reduce_mean(num1/den1 + num2/den2)

#======================================================================
class WeightedL1:
    def __init__(self, scale=50, nb_classes=128):
        """
        Parameters
        ----------

        """
        self.scale      = scale
        self.nb_classes = nb_classes
        x = np.linspace(0, self.nb_classes-1, self.nb_classes)
        wgt = np.exp(- x/self.scale).reshape([1,1,-1])
        self.wgt = float_me(wgt)

    def __call__(self, y_true, y_pred):
        """
        Parameters
        ----------
            - y_true : tf.Tensor, ground truths of shape=(B,N,nb_classes)
            - y_pred : tf.Tensor, predictions of shape=(B,N,nb_classes)

        Returns
        -------
        """
        diff = tf.math.abs(y_true-y_pred) * self.wgt
        return tf.reduce_mean(diff, axis=-1)

#======================================================================
def focal_crossentropy(y_true, y_pred, alpha=1.0, gamma=2.0, from_logits=False):
    """
    Implemention of the focal loss function from
    tfa.losses.SigmoidFocalCrossEntropy function.
    Focal loss was first introduced in the RetinaNet paper
    (https://arxiv.org/pdf/1708.02002.pdf). Focal loss is extremely useful for
    classification when you have highly imbalanced classes. It down-weights
    well-classified examples and focuses on hard examples. The loss value is
    much high for a sample which is misclassified by the classifier as compared
    to the loss value corresponding to a well-classified example. One of the
    best use-cases of focal loss is its usage in object detection where the
    imbalance between the background class and other classes is extremely high.

    Parameters
    ----------
        y_true: true targets tensor.
        y_pred: predictions tensor.
        alpha: balancing factor.
        gamma: modulating factor.

    Returns
    -------
        Weighted loss float `Tensor`. If `reduction` is `NONE`,this has the
        same shape as `y_true`; otherwise, it is scalar.
    """
    if gamma and gamma < 0:
        raise ValueError("Value of gamma should be greater than or equal to zero")

    # Get the cross_entropy for each entry
    ce = K.binary_crossentropy(y_true, y_pred, from_logits=from_logits)

    # If logits are provided then convert the predictions into probabilities
    if from_logits:
        pred_prob = tf.sigmoid(y_pred)
    else:
        pred_prob = y_pred

    p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
    alpha_factor = 1.0
    modulating_factor = 1.0

    if alpha:
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

    if gamma:
        modulating_factor = tf.pow((1.0 - p_t), gamma)

    # compute the final loss and return
    return tf.reduce_sum(alpha_factor * modulating_factor * ce, axis=-1)

#======================================================================
class WeightedCategoricalCrossEntropy(CategoricalCrossentropy):
    """ Implementation of Focal crossentropy.  """
    def __init__(self, nb_classes, scale, name='weight-xent', **kwargs):
        super().__init__(name=name, **kwargs)
        self.nb_classes = nb_classes
        self.scale      = scale
        x = np.linspace(0, self.nb_classes-1, self.nb_classes)
        wgt = np.exp(- x/self.scale).reshape([1,1,-1])
        self.wgt = float_me(wgt)

    #----------------------------------------------------------------------
    def call(self, y_true, y_pred):
        """
        Parameters
        ----------
            - y_true : tf.Tensor, ground truth tensor of shape=(B,N,nb_classes)
            - y_pred : tf.Tensor, output tensor of shape=(B,N,nb_classes)

        Returns
        -------
            tf.Tensor, loss tensor of shape=(B,N) if `reduction` is `NONE`,
            shape=() otherwise.
        """
        weighted_true = y_true * self.wgt
        return super().call(weighted_true, y_pred)

#======================================================================
class FocalCrossentropy(Loss):
    """ Implementation of Focal crossentropy.  """
    def __init__(self, from_logits=False, alpha=1.0, gamma=2.0,
                 reduction=Reduction.AUTO, name='focal_xent'):
        super().__init__(reduction=reduction, name=name)
        self.xent        = focal_crossentropy
        self.from_logits = from_logits
        self.alpha       = alpha
        self.gamma       = gamma

    #----------------------------------------------------------------------
    def call(self, y_true, y_pred):
        """
        Parameters
        ----------
            - y_true : tf.Tensor, ground truth tensor of shape=(B,N,nb_classes)
            - y_pred : tf.Tensor, output tensor of shape=(B,N,nb_classes)

        Returns
        -------
            tf.Tensor, loss tensor of shape=(B,N) if `reduction` is `NONE`,
            shape=() otherwise.
        """
        return self.xent(y_true, y_pred, self.alpha, self.gamma, self.from_logits)

#======================================================================
class CombinedLoss(Loss):
    """ Categorical crossentropy plus L1 on slice size.  """
    def __init__(self, from_logits=False, scale=50, nb_classes=128,
                 alpha=1.0, gamma=2.0, reduction=Reduction.AUTO,
                 name='xent-l1'):
        super().__init__(reduction=reduction, name=name)
        self.xent        = CategoricalCrossentropy(
                                from_logits=from_logits, name=name,
                                reduction=Reduction.NONE
                                )
        self.L1 = WeightedL1(scale=scale, nb_classes=nb_classes)

    #----------------------------------------------------------------------
    def call(self, y_true, y_pred):
        """
        Parameters
        ----------
            - y_true : tf.Tensor, ground truth tensor of shape=(B,N,nb_classes)
            - y_pred : tf.Tensor, output tensor of shape=(B,N,nb_classes)

        Returns
        -------
            tf.Tensor, loss tensor of shape=(B,N) if `reduction` is `NONE`,
            shape=() otherwise.
        """
        fxe = self.xent(y_true, y_pred)

        size_true = tf.reduce_sum(y_true, axis=1)
        size_pred = tf.reduce_sum(y_pred, axis=1)
        mae  = self.L1(size_true, size_pred)
        return  fxe + mae

#======================================================================
class CombinedFocalLoss(Loss):
    """ Focal crossentropy plus L1 on slice size.  """
    def __init__(self, from_logits=False, scale=50, nb_classes=128,
                 alpha=1.0, gamma=2.0, reduction=Reduction.AUTO,
                 name='focal_xent-l1'):
        super().__init__(reduction=reduction, name=name)
        self.xent        = focal_crossentropy
        self.from_logits = from_logits
        self.alpha       = alpha
        self.gamma       = gamma

        self.L1 = WeightedL1(scale=scale, nb_classes=nb_classes)

    #----------------------------------------------------------------------
    def call(self, y_true, y_pred):
        """
        Parameters
        ----------
            - y_true : tf.Tensor, ground truth tensor of shape=(B,N,nb_classes)
            - y_pred : tf.Tensor, output tensor of shape=(B,N,nb_classes)

        Returns
        -------
            tf.Tensor, loss tensor of shape=(B,N) if `reduction` is `NONE`,
            shape=() otherwise.
        """
        fxe = self.xent(y_true, y_pred, self.alpha, self.gamma, self.from_logits)

        size_true = tf.reduce_sum(y_true, axis=1)
        size_pred = tf.reduce_sum(y_pred, axis=1)
        mae  = self.L1(size_true, size_pred)
        return  fxe + mae

#======================================================================

def get_loss(setup, nb_classes):
    """
    Get the loss to train the model on:

    Parameters
    ---------
        - setup      : dict containing loss info
        - mb_classes : int, number of output classes

    Returns
    -------
        loss function wrapper
    """
    if setup.get('loss') == 'xent':
        from_logits = setup.get("from_logits")
        name        = "xent"
        return CategoricalCrossentropy(from_logits=from_logits, name=name)
    elif setup.get('loss') == 'focal':
        from_logits = setup.get("from_logits")
        gamma       = setup.get("gamma")
        name        = "focal_xent"
        return FocalCrossentropy(from_logits=from_logits, gamma=gamma, name=name)
    elif setup.get('loss') == 'xent_l1':
        from_logits = setup.get("from_logits")
        scale       = setup.get('wgt')
        name        = "xent-l1"
        return CombinedLoss(
                    from_logits=from_logits, scale=scale,
                    nb_classes=nb_classes, name=name
                    )
    elif setup.get('loss') == 'focal_l1':
        from_logits = setup.get("from_logits")
        gamma       = setup.get("gamma")
        scale       = setup.get('wgt')
        name        = "focal_xent-l1"
        return CombinedFocalLoss(
                    from_logits=from_logits, scale=scale,
                    nb_classes=nb_classes, gamma=gamma, name=name
                    )
    elif setup.get('loss') == 'w_xent':
        from_logits = setup.get("from_logits")
        scale       = setup.get('wgt')
        name        = "w_xent"
        return WeightedCategoricalCrossEntropy(
                    nb_classes=nb_classes, scale=scale,
                    from_logits=from_logits, name=name
                                              )
    else:
        raise NotImplementedError(f"loss named {setup.get('loss')} not implemented")