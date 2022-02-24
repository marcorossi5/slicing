# This file is part of SliceRL by M. Rossi
""" This module contains all the implemented custom layers. """
import tensorflow as tf
from tensorflow.keras.activations import sigmoid, tanh
from tensorflow.keras.layers import (
    Layer,
    Dense,
    MultiHeadAttention,
    LayerNormalization,
)
from slicerl.fast_attention.fast_attention import (
    SelfAttention,
    relu_kernel_transformation,
    softmax_kernel_transformation,
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
class SelfAttentionWrapper(SelfAttention):
    """
    A convenience wrapper for the SelfAttention layer.
    """

    def __init__(
        self,
        units,
        num_heads,
        attention_dropout=None,
        kernel_transformation="softmax",
        projection_matrix_type="random",
        nb_random_features=128,
        **kwargs,
    ):
        """
        Parameters
        ----------
            - units: int, output feature dimensionality
            - num_heads: int, number of heads in MultiHeadAttention layers
            - attention_dropout: float, dropout percentage
            - kernel_transformation: str, available options relu | softmax
            - projection_matrix_type: str, random or identity if None
            - nb_random_features: int, favor+ random features number

        Note
        ----
        The input feature dimensionality should be exactly divisible by the
        number of heads.
        """
        if kernel_transformation == "softmax":
            kt = softmax_kernel_transformation
        elif kernel_transformation == "relu":
            kt = relu_kernel_transformation
        else:
            raise NotImplementedError(
                "Kernel transformation not implemented"
                f", found {kernel_transformation}"
            )

        super(SelfAttentionWrapper, self).__init__(
            units,
            num_heads,
            attention_dropout,
            kernel_transformation=kt,
            projection_matrix_type=projection_matrix_type,
            nb_random_features=nb_random_features,
            **kwargs,
        )

    def call(self, query_input, bias=None, training=None, **kwargs):
        return super(SelfAttentionWrapper, self).call(
            query_input, bias, training, **kwargs
        )


class TransformerEncoder(Layer):
    """
    Implementation of ViT Encoder layer. This block exploits the fast
    implementation of the Attention mechanism for better memory management.
    """

    def __init__(
        self,
        units,
        mha_heads,
        attention_type,
        kernel_transformation="softmax",
        projection_matrix_type="random",
        nb_random_features=128,
        **kwargs,
    ):
        """
        Parameters
        ----------
            - units: int, output feature dimensionality
            - mha_heads: int, number of heads in MultiHeadAttention layers
            - attention_type: str, available options original | favor+
            - kernel_transformation: str, available options relu | softmax
            - projection_matrix_type: str, random or identity if None
            - nb_random_features: int, favor+ random features number
        """
        super(TransformerEncoder, self).__init__(**kwargs)
        self.units = units
        self.mha_heads = mha_heads
        self.attention_type = attention_type
        self.kernel_transformation = kernel_transformation
        self.projection_matrix_type = projection_matrix_type
        self.nb_random_features = nb_random_features

        self.norm0 = LayerNormalization(axis=-1, name="ln_0")

        self.fc0 = Dense(units, activation="relu", name="mlp_0")
        self.fc1 = Dense(units, activation="relu", name="mlp_1")

        # self.conv = Conv1D(units, name="conv")
        self.norm1 = LayerNormalization(axis=-1, name="ln_1")

    # ----------------------------------------------------------------------
    def build(self, input_shape):
        """
        Parameters
        ----------
        """
        units = input_shape[-1]
        if self.attention_type == "original":
            self.mha = MultiHeadAttention(self.mha_heads, units, name="mha")
        elif self.attention_type == "favor+":
            self.mha = SelfAttentionWrapper(
                units,
                self.mha_heads,
                kernel_transformation=self.kernel_transformation,
                projection_matrix_type=self.projection_matrix_type,
                nb_random_features=self.nb_random_features,
                name="mha",
            )
        else:
            raise NotImplementedError(
                f"Attention type {self.attention_type} not implemented"
            )
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
        if self.attention_type == "original":
            x += self.mha(x, x)
        else:
            x += self.mha(x)
        # layer normalization
        x = self.norm0(x)
        x = self.fc1(self.fc0(x))
        output = self.norm1(x)
        return output

    # ----------------------------------------------------------------------
    def get_config(self):
        return {"units": self.units, "mha_heads": self.mha_heads}


# ======================================================================
# Feed-forward layers
class Head(Layer):
    """Implementation of stacking of feed-forward layers."""

    def __init__(
        self,
        filters,
        dropout_idxs=None,
        dropout=None,
        activation="relu",
        kernel_initializer="GlorotUniform",
        name="head",
        **kwargs,
    ):
        """
        Parameters
        ----------
            - filters: list, the number of filters for each dense layer

        """
        super(Head, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.dropout_idxs = dropout_idxs
        self.dropout = dropout
        self.activation = activation
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        lyrs = []

        for i, filters in enumerate(self.filters):
            lyrs.append(
                Dense(
                    filters,
                    activation=self.activation,
                    kernel_initializer=self.kernel_initializer,
                    name=f"dense_{i}",
                )
            )

            # if i in self.dropout_idxs:
            #     pass
            #     # lyrs.append(BatchNormalization(name=f"bn_{self.nb_head}_{i}"))
            #     # lyrs.append(Dropout(self.dropout, name=f"do_{self.nb_head}_{i}"))

        self.fc = lyrs

    # ----------------------------------------------------------------------
    def call(self, x):
        """
        Layer forward pass.

        Parameters
        ----------
            - x : list of input tf.Tensors
        Returns
        -------
            - tf.Tensor of shape=(B,N,K,do)
        """
        for l in self.fc:
            x = l(x)
        return x

    # ----------------------------------------------------------------------
    def get_config(self):
        config = super(Head, self).get_config()
        config.update(
            {
                "dropout": self.dropout,
                "activation": self.activation,
            }
        )
        return config
