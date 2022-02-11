# This file is part of SliceRL by M. Rossi
from slicerl.AbstractNet import AbstractNet
from slicerl.FFNN import Head
import tensorflow as tf
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import (
    Dense,
    Conv1D,
    BatchNormalization,
    Concatenate,
    MultiHeadAttention,
    LeakyReLU,
)
from tensorflow.keras.activations import sigmoid


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

        # self.mha_filters = [16, 32, 64, 128]
        self.mha_filters = [256,]
        self.mhas = []
        for ih, dout in enumerate(self.mha_filters):
            self.mhas.append(
                MultiHeadAttention(
                    self.mha_heads, dout, output_shape=dout, name=f"mha_{ih}"
                )
            )

        # self.cumulative_gradients = None
        self.cumulative_counter = tf.constant(0)

        # store some useful parameters for the fc layers
        # self.fc_filters = [64, 32, 16, 8, 4, 2, 1]
        self.fc_filters = [128, 32, 8, 1]
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
            x = self.activation(mha(x, x))

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
