# This file is part of SliceRL by M. Rossi
from slicerl.AbstractNet import AbstractNet
from slicerl.FFNN import Head
import tensorflow as tf
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import (
    Layer,
    Dense,
    Conv1D,
    BatchNormalization,
    Concatenate,
    MultiHeadAttention,
    LeakyReLU,
    LayerNormalization,
)
from tensorflow.keras.activations import sigmoid, tanh


def add_extension(name):
    """
    Adds extension to cumulative variables while looping over it.

    Parameters
    ----------
        - name: str, the weight name to be extended

    Returns
    -------
        - str, the extended weight name
    """
    ext = "_cum"
    l = name.split(":")
    l[-2] += ext
    return ":".join(l[:-1])


# ======================================================================
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

    def get_config(self):
        return {"units": self.units, "mha_heads": self.mha_heads}


# ======================================================================
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

    def build(self, input_shape):
        """
        Parameters
        ----------
        """
        units = input_shape[-1]
        self.mha = MultiHeadAttention(self.mha_heads, units, name="mha")
        super(TransformerEncoder, self).build(input_shape)

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

    def get_config(self):
        return {"units": self.units, "mha_heads": self.mha_heads}


# ======================================================================
class CMNet(AbstractNet):
    """Class deifining CM-Net: Cluster merging network."""

    def __init__(
        self,
        batch_size=1,
        f_dims=2,
        fc_heads=5,
        mha_heads=5,
        activation="relu",
        use_bias=True,
        verbose=False,
        name="CM-Net",
        **kwargs,
    ):
        """
        Parameters
        ----------
            - f_dims     : int, point cloud feature dimensions
            - activation : str, default layer activation
            - use_bias   : bool, wether to use bias or not
        """
        super(CMNet, self).__init__(f_dims=f_dims, name=name, **kwargs)

        # store args
        self.batch_size = int(batch_size)
        self.f_dims = f_dims
        self.fc_heads = fc_heads
        self.mha_heads = mha_heads
        self.activation = activation
        self.use_bias = use_bias
        self.verbose = verbose

        self.mha_filters = [64, 128]
        self.mhas = []
        # self.mha_ifilters = [self.fc_heads] + self.mha_filters[:-1]
        for ih, dout in enumerate(self.mha_filters):
            # self.mhas.append(MyGRU(dout, self.mha_heads, name=f"mha_{ih}"))
            self.mhas.append(TransformerEncoder(dout, self.mha_heads, name=f"mha_{ih}"))

        # self.cumulative_gradients = None
        self.cumulative_counter = tf.constant(0)

        # store some useful parameters for the fc layers
        self.fc_filters = [64, 32, 16, 8, 4, 2, 1]
        # self.dropout_idxs = [0, 1]
        self.dropout_idxs = []
        self.dropout = 0.2
        self.heads = []
        for ih in range(self.fc_heads):
            self.heads.append(
                Head(
                    self.fc_filters,
                    ih,
                    self.dropout_idxs,
                    self.dropout,
                    activation=self.activation,
                    name=f"head_{ih}",
                )
            )
        self.concat = Concatenate(axis=-1, name="cat")

        self.build()

        self.cumulative_gradients = [
            tf.Variable(
                tf.zeros_like(this_var),
                trainable=False,
                name=add_extension(this_var.name),
            )
            for this_var in self.trainable_variables
        ]
        self.cumulative_counter = tf.Variable(
            tf.constant(0), trainable=False, name="cum_counter"
        )

    # ----------------------------------------------------------------------
    def build(self):
        """Explicitly build the weights."""
        super(CMNet, self).build((None, None, self.f_dims))

    # ----------------------------------------------------------------------
    def call(self, inputs):
        """
        Parameters
        ----------
            - inputs : list of tf.Tensors, point cloud of shape=(B,N,dims) and
                       feature point cloud of shape=(B,N,f_dims)

        Returns
        -------
            tf.Tensor, output tensor of shape=(B,N,nb_classes)
        """
        x = inputs
        # x = tf.expand_dims(inputs, 0)
        for mha in self.mhas:
            x = self.activation(mha(x))

        x = tf.reduce_max(x, axis=1)

        results = []
        for head in self.heads:
            results.append(head(x))

        return tf.reduce_mean(sigmoid(self.concat(results)), axis=-1)

    # ----------------------------------------------------------------------
    def model(self):
        inputs = Input(shape=(None, self.f_dims), name="pc")
        return Model(inputs=inputs, outputs=self.call(inputs), name=self.name)

    # ----------------------------------------------------------------------
    def reset_cumulator(self):
        """
        Reset counter and gradients cumulator gradients.
        """

        for i in range(len(self.cumulative_gradients)):
            self.cumulative_gradients[i].assign(
                tf.zeros_like(self.trainable_variables[i])
            )
        self.cumulative_counter.assign(tf.constant(1))

    # ----------------------------------------------------------------------
    def increment_counter(self):
        """
        Reset counter and gradients cumulator gradients.
        """
        self.cumulative_counter.assign_add(tf.constant(1))

    # ----------------------------------------------------------------------
    def train_step(self, data):
        """
        The network accumulates the gradients according to batch size, to allow
        gradient averaging over multiple inputs. The aim is to reduce the loss
        function fluctuations.
        """
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        if self.verbose:
            reset_2 = (self.cumulative_counter + 1) % self.batch_size == 0
            tf.cond(
                reset_2,
                lambda: print_loss(x, y, y_pred, loss),
                lambda: False,
            )

        reset = self.cumulative_counter % self.batch_size == 0
        tf.cond(reset, self.reset_cumulator, self.increment_counter)

        # Compute gradients
        trainable_vars = self.trainable_variables

        gradients = tape.gradient(loss, trainable_vars)
        for i, grad in enumerate(gradients):
            self.cumulative_gradients[i].assign_add(grad / self.batch_size)

        # Update weights
        reset = self.cumulative_counter % self.batch_size == 0

        if self.verbose:
            reset_1 = (self.cumulative_counter - 1) % self.batch_size == 0
            tf.cond(
                reset_1,
                lambda: print_gradients(zip(self.cumulative_gradients, trainable_vars)),
                lambda: False,
            )

        tf.cond(
            reset,
            lambda: self.optimizer.apply_gradients(
                zip(self.cumulative_gradients, trainable_vars)
            ),
            lambda: False,
        )

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


def print_loss(x, y, y_pred, loss):
    # tf.print(", x:", tf.reduce_mean(x), tf.math.reduce_std(x), ", y:", y, ", y_pred:", y_pred, ", loss:", loss)
    tf.print(y, y_pred)
    return True


def print_gradients(gradients):
    for g, t in gradients:
        tf.print(
            "Param:",
            g.name,
            ", value:",
            tf.reduce_mean(t),
            tf.math.reduce_std(t),
            ", grad:",
            tf.reduce_mean(g),
            tf.math.reduce_std(g),
        )
    tf.print("---------------------------")
    return True
