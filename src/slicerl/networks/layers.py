# This file is part of SliceRL by M. Rossi
""" This module contains all the implemented custom layers. """
import tensorflow as tf
from tensroflow.keras import Sequential
from tensorflow.keras.activations import sigmoid, tanh
from tensorflow.keras.layers import (
    Layer,
    Dense,
    MultiHeadAttention,
    LayerNormalization,
)

# General-op layers
class ReduceMax(Layer):
    """Implementation of Reduce Max layer."""

    def __init__(self, axis=-1, **kwargs):
        """
        Parameters
        ----------
            - axis: the axis to reduce. Default is the last axis
        """
        super(ReduceMax, self).__init__(**kwargs)
        self.axis = axis
        self.op = lambda x: tf.reduce_max(x, axis=self.axis)

    # ----------------------------------------------------------------------
    def call(self, x):
        return self.op(x)

    # ----------------------------------------------------------------------
    def get_config(self):
        return {"axis": self.axis}


# ======================================================================
# Recurrent layers
class MyGRU(Layer):
    """Implementation of the GRU layer."""

    def __init__(self, units, mha_heads, **kwargs):
        """
        Parameters
        ----------
            - units: int, output feature dimensionality
        """
        super(MyGRU, self).__init__(**kwargs)
        self.units = units
        self.wr = Dense(units, name="wr", activation=None)
        self.hr = Dense(units, name="hr", activation=None, use_bias=False)

        self.wz = Dense(units, name="wz", activation=None)
        self.hz = Dense(units, name="hz", activation=None, use_bias=False)

        self.wn = Dense(units, name="wn", activation=None)
        self.hn = Dense(units, name="hn", activation=None)

        self.act = sigmoid
        self.n_act = tanh

        self.mha_heads = mha_heads
        self.mha = MultiHeadAttention(
            self.mha_heads, units, output_shape=units, name="mha"
        )

    # ----------------------------------------------------------------------
    def call(self, x):
        """
        Parameters
        ----------
            - x: tf.Tensor, input tensor of shape=(B, L, d_in)

        Returns
        -------
            - tf.Tensor, output tensor of shape=(B, L, d_out)
        """
        h = self.mha(x, x)
        rt = self.act(self.wr(x) + self.hr(h))
        zt = self.act(self.wz(x) + self.hz(h))
        nt = self.n_act(self.wn(x) + rt * self.hn(h))
        return (1 - zt) * nt + zt * h

    # ----------------------------------------------------------------------
    def get_config(self):
        return {"units": self.units, "mha_heads": self.mha_heads}


# ======================================================================
# Attention layers
class TransformerEncoder(Layer):
    """Implementation of ViT Encoder layer."""

    def __init__(self, units, mha_heads, **kwargs):
        """
        Parameters
        ----------
            - units: int, output feature dimensionality
            - mha_heads: int, number of heads in MultiHeadAttention layers
        """
        super(TransformerEncoder, self).__init__(**kwargs)
        self.units = units
        self.mha_heads = mha_heads

        self.norm0 = LayerNormalization(axis=-1, name="ln_0")

        self.mlp = Sequential(
            [Dense(units, activation="relu"), Dense(units, activation=None)], name="mlp"
        )

        # self.conv = Conv1D(units, name="conv")
        self.norm1 = LayerNormalization(axis=-1, name="ln_1")

    # ----------------------------------------------------------------------
    def build(self, input_shape):
        """
        Parameters
        ----------
        """
        units = input_shape[-1]
        self.mha = MultiHeadAttention(self.mha_heads, units, name="mha")
        super(TransformerEncoder, self).build(input_shape)

    # ----------------------------------------------------------------------
    def call(self, x):
        """
        Parameters
        ----------
            - x: tf.Tensor, input tensor of shape=(B, L, d_in)

        Returns
        -------
            - tf.Tensor, output tensor of shape=(B, L, d_out)
        """
        # residual and multi head attention
        x += self.mha(x, x)
        # layer normalization
        x = self.norm0(x)
        return self.norm1(self.mlp(x))

    # ----------------------------------------------------------------------
    def get_config(self):
        return {"units": self.units, "mha_heads": self.mha_heads}


# ======================================================================
# Feed-forward layers
class Head(Layer):
    """Class defining Spatial Encoding Attention Convolutional Layer."""

    def __init__(
        self,
        filters,
        nb_head,
        dropout_idxs,
        dropout,
        activation="relu",
        name="head",
        **kwargs,
    ):
        super(Head, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.nb_head = nb_head
        self.dropout_idxs = dropout_idxs
        self.dropout = dropout
        self.activation = activation
        lyrs = []

        for i, filters in enumerate(self.filters):
            lyrs.append(
                Dense(
                    filters,
                    activation=self.activation,
                    name=f"dense_{self.nb_head}_{i}",
                )
            )

            if i in self.dropout_idxs:
                pass
                # lyrs.append(BatchNormalization(name=f"bn_{self.nb_head}_{i}"))
                # lyrs.append(Dropout(self.dropout, name=f"do_{self.nb_head}_{i}"))

        self.fc = Sequential(lyrs, name=name)

    # ----------------------------------------------------------------------
    def call(self, inputs):
        """
        Layer forward pass.

        Parameters
        ----------
            - inputs : list of tf.Tensors, tensors describing spatial and
                       feature point cloud of shapes=[(B,N,dims), (B,N,f_dims)]

        Returns
        -------
            tf.Tensor of shape=(B,N,K,do)
        """
        return self.fc(inputs)

    # ----------------------------------------------------------------------
    def get_config(self):
        config = super(Head, self).get_config()
        config.update(
            {
                "nb_head": self.nb_head,
                "dropout": self.dropout,
                "activation": self.activation,
            }
        )
        return config