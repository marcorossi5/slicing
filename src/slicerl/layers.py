# This file is part of SliceRL by M. Rossi
# Inlcudes the implementation of the AttentionLayer
import tensorflow as tf
from tensorflow import matmul
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Permute
from tensorflow.keras.activations import softmax

# ======================================================================
class Attention(Model):
    """ Class defining Attention Layer. """
    # ----------------------------------------------------------------------
    def __init__(
        self,
        units,
        h_hunits,
        activation="relu",
        use_bias=True,
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
        super(Attention, self).__init__(**kwargs)
        self.units = units
        self.h_units = h_hunits
        self.activation = activation
        self.use_bias = use_bias
        self.fc_query = Dense(
            self.h_units,
            activation=self.activation,
            use_bias=self.use_bias,
            name="fc_query",
        )
        self.fc_key = Dense(
            self.h_units,
            activation=self.activation,
            use_bias=self.use_bias,
            name="fc_key",
        )
        self.fc_value = Dense(
            self.h_units,
            activation=self.activation,
            use_bias=self.use_bias,
            name="fc_value",
        )
        self.fc_mixing = Dense(
            self.units,
            activation=self.activation,
            use_bias=self.use_bias,
            name="fc_mixing",
        )

        self.perm = Permute((2,1))

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
        value = self.fc_query(inputs)
        kq = softmax(matmul(key, query, transpose_b=True), axis=-1)
        attention = self.perm(matmul(value, kq, transpose_a=True))
        return self.fc_mixing(attention)

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
