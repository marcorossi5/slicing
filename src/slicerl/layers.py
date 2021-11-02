# This file is part of SliceRL by M. Rossi
# Inlcudes the implementation of the AttentionLayer
import tensorflow as tf
from tensorflow import matmul
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, Permute, TimeDistributed, Concatenate
from tensorflow.keras.activations import softmax
from slicerl.config import float_me

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
        V = self.fc_query(inputs)
        kq = Q*K/tf.math.sqrt(self.sqrtdk)
        
        weights = tf.exp(kq) / tf.reduce_sum(tf.exp(kq), axis=-1, keepdims=True)

        return V*weights

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
        self.cat = Concatenate(axis=-1, name='cat')
    
    def __call__(self, inputs):
        global_max = tf.math.reduce_max(inputs, axis=-2, name='max')
        global_min = tf.math.reduce_min(inputs, axis=-2, name='min')
        global_avg = tf.math.reduce_mean(inputs, axis=-2, name='avg')
        return self.cat([global_max, global_min, global_avg])
