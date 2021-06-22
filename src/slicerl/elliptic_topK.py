import tensorflow as tf
import numpy as np


def RotoTranslation(points, translations, rotations):
    """
    Apply translation and rotation to find elliptical distances.


    Parameters
    ----------
        - points: tf.Tensor, points of shape=(num points, 2)
        - translations: tf.Tensor, translation parameters of shape=(num queries,2)
        - rotations: tf.Tensor, translation parameters of shape=(num queries,2)

    Returns
    -------
        - tf.Tensor, of shape=(num queries, num points, 2)
    """
    # translate
    trans = tf.expand_dims(points, 0) - tf.expand_dims(translations, 1)

    x = trans[..., 0]
    y = trans[..., 1]

    # rotate
    cost = rotations[:, :1]
    sint = rotations[:, 1:]
    xrot = cost * x + sint * y
    yrot = cost * y - sint * x

    return tf.stack([xrot, yrot], axis=-1)


class EllipticTopK:
    def __init__(self, a, b, K, batched=False):
        # TODO: split queries into sequential computation to save memory
        # normalize axes
        self.majAxis = a if a >= b else b
        self.minAxis = b if a >= b else a
        self.K = K
        self.batched = batched

    def __call__(self, points, queries):
        """
        Parameters
        ----------
            - points: tf.Tensor, of shape=(n points, 4)
            - queries: tf.Tensor, of shape=(n queries, 4)
        """
        if self.batched:
            points = points[0]
            queries = queries[0]
        points = points[..., :2]

        # translation parameters
        centers = queries[..., :2]

        # rotation parameters
        exp_dir = queries[..., 2:]
        norm = tf.norm(exp_dir, ord="euclidean", axis=-1, keepdims=True)
        exp_dir = tf.where(norm != 0, exp_dir / norm, exp_dir + 1)

        rot = RotoTranslation(points, centers, exp_dir)

        x = rot[..., 0] / self.majAxis
        y = rot[..., 1] / self.minAxis

        dists = -(x ** 2) - y ** 2
        knn_idx = tf.math.top_k(dists, k=self.K, sorted=True)

        if self.batched:
            return tf.expand_dims(knn_idx.indices, 0)
        return knn_idx.indices
