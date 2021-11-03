# This file is part of SliceRL by M. Rossi
# Inlcudes the implementation of the AttentionLayer
import tensorflow as tf
from tensorflow import matmul
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, Concatenate
from slicerl.config import float_me, int_me, TF_DTYPE, TF_DTYPE_INT

# ======================================================================
class Attention(Model):
    """ Class defining Attention Layer. """

    # ----------------------------------------------------------------------
    def __init__(
        self,
        iunits,
        units,
        activation="relu",
        use_bias=True,
        **kwargs,
    ):
        """
        Parameters
        ----------
            - iunits: int, input channels
            - units: int, output channels
            - activation: str, MLP layer activation
            - use_bias: bool, wether to use bias in MLPs
        """
        super(Attention, self).__init__(**kwargs)
        self.units = units
        self.iunits = iunits
        self.sqrtdk = tf.math.sqrt(float_me(iunits))
        self.activation = activation
        self.use_bias = use_bias
        self.fc_query = Dense(
            self.units,
            activation=self.activation,
            use_bias=self.use_bias,
            name="fc_query",
        )
        self.fc_key = Dense(
            self.units,
            activation=self.activation,
            use_bias=self.use_bias,
            name="fc_key",
        )
        self.fc_value = Dense(
            self.units,
            activation=self.activation,
            use_bias=self.use_bias,
            name="fc_value",
        )

    # ----------------------------------------------------------------------
    def call(self, inputs):
        """
        Layer forward pass.

        Parameters
        ----------
            - inputs: tf.Tensors, tensors describing the clusters pair point
                      cloud of shape=(B,N,Cin)

        Returns
        -------
            - tf.Tensor of shape=(B,N,Cout)
        """
        Q = self.fc_query(inputs)
        K = self.fc_key(inputs)
        V = self.fc_value(inputs)
        qk = ragged_mult(tf.stack([Q,K], axis=1))/self.sqrtdk
        wgts = tf.exp(qk) / tf.reduce_sum(tf.exp(qk), axis=-1, keepdims=True)
        return matmul(wgts, V)
    # ----------------------------------------------------------------------
    def get_config(self):
        config = super(Attention, self).get_config()
        config.update(
            {
                "units": self.units,
                "activation": self.activation,
                "use_bias": self.use_bias,
            }
        )
        return config


# ======================================================================
class GlobalFeatureExtractor(Layer):
    def __init__(self, **kwargs):
        super(GlobalFeatureExtractor, self).__init__(**kwargs)
        self.cat = Concatenate(axis=-1, name="cat")

    def __call__(self, inputs):
        global_max = tf.math.reduce_max(inputs, axis=-2, name="max")
        global_min = tf.math.reduce_min(inputs, axis=-2, name="min")
        global_avg = tf.math.reduce_mean(inputs, axis=-2, name="avg")
        return self.cat([global_max, global_min, global_avg])


# ======================================================================
def map_op(x):
    """
    Operation that runs on each element of the first axis or the rt argumetn of
    ragged_mult function. The first axis of the x tensor actually has dimension
    2.

    Parameters
    ----------
        - x: tf.RaggedTensor, of shape=(None,None,C)

    Returns
    -------
        - tf.RaggedTensor, of shape [None,None]
    """
    out = tf.cast(tf.matmul(x[0], x[1], transpose_b=True), dtype=TF_DTYPE)
    N = int_me(tf.shape(x[0])[0])
    splits = tf.concat([tf.zeros([1], dtype=TF_DTYPE_INT), tf.fill([N], N)], axis=0)
    splits = tf.math.cumsum(splits)
    return tf.RaggedTensor.from_row_splits(
        values=tf.reshape(out, [-1]), row_splits=splits
    )


# ======================================================================
def ragged_mult(rt):
    """
    Takes a tf.RaggedTensor rt of shape [B,2,(N),C], where second axis represents
    a and b tensors of the same shape [(N),C], and computes the a.T*b operation
    resulting in a tf.RaggedTensor of shape [B, (N), (N)].

    Parameters
    ----------
        - rt: tf.RaggedTensor; input tensor of shape=[B,2,(N),C]

    Returns
    -------
        - tf.RaggedTensor, output tensor of shape [B,(N),(N)]
    """
    return tf.map_fn(
        map_op,
        rt,
        fn_output_signature=tf.RaggedTensorSpec(
            shape=[None, None], dtype=TF_DTYPE, row_splits_dtype=TF_DTYPE_INT
        ),
    )
