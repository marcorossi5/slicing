# This file is part of SliceRL by M. Rossi
# Inlcudes the implementation of the AttentionLayer
import tensorflow as tf
from tensorflow import matmul
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Concatenate, Layer
from tensorflow.keras.activations import softmax

# ======================================================================
class Attention(Layer):
    """ Class defining Attention Layer. """
    # ----------------------------------------------------------------------
    def __init__(
        self,
        units,
        h_hunits,
        activation="relu",
        use_bias=True,
        name=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
            - units: int, output channels
            - h_units: int, hidden channels
            - activation: str, MLP layer activation
            - use_bias: bool, wether to use bias in MLPs
        """
        super(Attention, self).__init__(name=name, **kwargs)
        self.units = units
        self.h_units = h_hunits
        self.activation = activation
        self.use_bias = use_bias
        self.fc_query = Dense(
            self.h_units,
            activation="tanh",
            use_bias=self.use_bias,
            name="fc_query",
        )
        self.fc_key = Dense(
            self.h_units,
            activation="tanh",
            use_bias=self.use_bias,
            name="fc_key",
        )
        self.fc_value = Dense(
            self.h_units,
            activation=self.activation,
            use_bias=self.use_bias,
            name="fc_value",
        )

    def build(self, input_shape):
        with tf.name_scope("fc_query"):
            self.fc_key.build(input_shape)
        with tf.name_scope("fc_key"):
            self.fc_query.build(input_shape)
        with tf.name_scope("fc_value"):
            self.fc_value.build(input_shape)

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
        query = self.fc_query(inputs)
        key = self.fc_key(inputs)
        value = self.fc_value(inputs)
        qk = softmax(matmul(query, key, transpose_b=True), axis=-1)
        # qk = qk / tf.reduce_max(qk, axis=-1, keepdims=True)
        # tf.print("QK.t", tf.reduce_min(qk), tf.reduce_mean(qk), tf.reduce_max(qk))
        attention = matmul(qk, value)
        # tf.print(self.name, "Inputs: ", tf.reduce_min(inputs), tf.reduce_max(inputs), "att: ", tf.reduce_min(attention), tf.reduce_max(attention))
        return attention

    # ----------------------------------------------------------------------
    def get_config(self):
        config = super(Attention, self).get_config()
        config.update(
            {
                "units": self.units,
                "h_units": self.h_units,
                "activation": self.activation,
                "use_bias": self.use_bias,
            }
        )
        return config

# ======================================================================
class GlobalFeatureExtractor(Layer):
    def __init__(self, name=None, **kwargs):
        super(GlobalFeatureExtractor, self).__init__(name=name, **kwargs)
        self.cat = Concatenate(axis=-1, name="cat")

    def __call__(self, inputs):
        global_max = tf.math.reduce_max(inputs, axis=-2, name="max", keepdims=True)
        global_min = tf.math.reduce_min(inputs, axis=-2, name="min", keepdims=True)
        global_avg = tf.math.reduce_mean(inputs, axis=-2, name="avg", keepdims=True)
        # return global_avg
        return self.cat([global_max, global_min, global_avg])