# This file is part of SliceRL by M. Rossi
""" Module containing utility functions for computations. """
from slicerl.utils.configflow import EPS, EPS_TF, float_me
from collections import deque
import numpy as np
import tensorflow as tf


# ======================================================================
def get_window_width(masses, lower_frac=20, upper_frac=80):
    """Returns"""
    lower = np.nanpercentile(masses, lower_frac)
    upper = np.nanpercentile(masses, upper_frac)
    median = np.median(masses[(masses > lower) & (masses < upper)])
    return lower, upper, median


# ======================================================================
def confusion_matrix_per_event(y_true, y_pred):
    """
    Computes confusion matrix values for each graph in list

    Parameters
    ----------
        - y_true : list, of ground truths graphs arrays each of shape=(N,1+K)
        - y_true : list, of predicted graphs arrays each of shape=(N,1+K)

    Returns
      - tp      true positives ratio
      - fp      false positives ratio
      - fn      false negatives ratio
      - tn      true negatives ratio
    """
    # flatten arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    tp = []
    fp = []
    fn = []
    tn = []
    for preds, truths in zip(y_true, y_pred):
        truths = truths.astype(bool)
        tot = truths.shape[0]
        tp.append(len(np.where(preds[truths] == 1)[0]) / tot)
        fp.append(len(np.where(preds[~truths] == 1)[0]) / tot)
        fn.append(len(np.where(preds[truths] == 0)[0]) / tot)
        tn.append(len(np.where(preds[~truths] == 0)[0]) / tot)

    return np.array(tp), np.array(fp), np.array(fn), np.array(tn)


# ======================================================================
def m_lin_fit(x, y):
    """Compute the angular coefficient of a linear fit."""
    assert x.shape == y.shape
    n = x.shape[0]
    num = n * (x * y).sum() - x.sum() * y.sum()
    den = n * (x * x).sum() - (x.sum()) ** 2
    return num / (den + EPS)


# ======================================================================
def pearson_distance(x, y, axis):
    """Computes modified pearson distance."""
    xc = x - x.mean()
    yc = y - y.mean()
    num = (xc * yc).sum()
    den = (xc**2).sum() * (yc**2).sum()


# ======================================================================
def rsum(x, axis=None, keepdims=False):
    return tf.reduce_sum(x, axis=axis, keepdims=keepdims)


# ======================================================================
def rmean(x, axis=None, keepdims=False):
    return tf.reduce_mean(x, axis=axis, keepdims=keepdims)


# ======================================================================
def m_lin_fit_tf(pc):
    """
    Compute the angular coefficient of a linear fit.

    Parameters
    ----------
        - pc: tf.Tensor, spatial point cloud of shape=(B,N,K,dims)

    Returns
    -------
        - tf.Tensor, squared pearson coefficient of shape=(B,N,1)
    """
    x = pc[:, :, :, :1]
    y = pc[:, :, :, 1:]
    n = float_me(tf.shape(x)[-2])
    num = n * rsum(x * y, -2, True) - rsum(x, -2, True) * rsum(y, -2, True)
    den = n * rsum(x * x, -2, True) - rsum(x, -2, True) ** 2
    return num / (den + EPS_TF)


# ======================================================================
def pearson_distance_tf(pc):
    """
    Computes modified pearson distance.

    Parameters
    ----------
        - pc: tf.Tensor, spatial point cloud of shape=(B,N,K,dims)

    Returns
    -------
        - tf.Tensor, squared pearson coefficient of shape=(B,N,1)
    """
    x = pc[:, :, :, :1]
    y = pc[:, :, :, 1:]
    xc = x - rmean(x, -2, True)
    yc = y - rmean(y, -2, True)
    num = rsum(xc * yc, -2, True)
    den = rsum(xc**2, -2, True) * rsum(yc**2, -2, True)
    return 1 - num**2 / (den + EPS)


# ======================================================================
def mse(x, y):
    return ((x - y) ** 2).mean()


# ======================================================================
def bce_loss(x, y):
    # Warning: computing log is expensive
    ratio = 0.1  # percentage of ones over zeros
    loss = -y * np.log(x + EPS) / ratio - (1 - y) * np.log(1 - x + EPS) / (1 - ratio)
    return loss.mean()


# ======================================================================
def dice_loss(x, y):
    ix = 1 - x
    iy = 1 - y
    num1 = (x * y).sum(-1) + EPS
    den1 = (x * x + y * y).sum(-1) + EPS
    num2 = (ix * iy).sum(-1) + EPS
    den2 = (ix * ix + iy * iy).sum(-1) + EPS
    return 1 - (num1 / den1 + num2 / den2).mean()


# ======================================================================
def efficiency_rejection_rate_loss(x, y):
    efficiency = np.count_nonzero(x[y]) / (np.count_nonzero(y) + EPS)
    rejection_rate = np.count_nonzero(x[~y]) / (np.count_nonzero(~y) + EPS)
    return 1 - efficiency + rejection_rate


# ======================================================================
def onehot(ind, depth):
    """
    One-hot encoding on the last axis

    Parameters
    ----------
        - ind   : np.array, array of indices of int dtype
        - depth : int, length of the one-hot encoding axis

    Returns
    -------
        np.array, one-hot encoded array of shape=(ind.shape + (depth,))
    """
    return np.eye(depth)[ind.astype(np.int16)]


# ======================================================================
def onehot_to_indices(onehot):
    """
    From one-hot encoding to indices on the last axis

    Parameters
    ----------
        - one_hot : np.array, array of one-hot encoded

    Returns
    -------
        np.array, indices array of shape=(one_hot.shape[:-1])
    """
    return np.argmax(onehot, axis=-1)


# ======================================================================
def dfs(visited, node, graph):
    """
    Depth First Search graph traversing.

    Problem of this implementation is that instantiated for loops are already given
    and cannot be modified. Some redundancy occurs in this implementation.
    """
    if node not in visited:
        visited.add(node)
        print(f"Check in node {node}, slice: {visited}")
        to_visit = graph[node].difference(visited)
        for neighbor in to_visit:
            print(f"For loop made with {to_visit}")
            dfs(visited, neighbor, graph)


# ======================================================================
def bfs(sl, visited, root, graph):
    """
    Breadth First Search graph traversing. Fills in the visited set with the
    node indices reachable from root node.

    Parameters
    ----------
        - sl   : set, of points in the same slice
        - visited : set, of already visited nodes
        - root    : int, index of root search node
        - graph   : list, of same slice neighbours indices
    """
    queue = deque([root])
    sl.add(root)
    visited.add(root)

    while queue:

        # Dequeue a node from queue
        node = queue.popleft()
        # print(str(node) + " ", end="")

        # If not visited, mark it as visited, and
        # enqueue it
        for neighbour in graph[node]:
            if (neighbour not in visited) and (neighbour not in sl):
                sl.add(neighbour)
                visited.add(neighbour)
                queue.append(neighbour)
