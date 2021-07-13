# This file is part of SliceRL by M. Rossi
from slicerl.layers import AbstractNet
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, BatchNormalization

# ======================================================================
class CMNet(AbstractNet):
    """ Class deifining CM-Net: Cluster merging network. """

    def __init__(
        self,
        batch_size=1,
        f_dims=2,
        activation="relu",
        use_bias=True,
        name="SEAC-Net",
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
        self.activation = activation
        self.use_bias = use_bias

        # self.cumulative_gradients = None
        self.cumulative_counter = tf.constant(0)

        # store some useful parameters
        self.nb_filters = [32, 64, 128, 256, 128, 64, 32, 16, 8]
        kernel_sizes = [2] + [1] * (len(self.nb_filters) - 1)
        self.convs = [
            Conv1D(
                filters,
                kernel_size,
                # kernel_constraint=MaxNorm(axis=[0, 1]),
                activation=self.activation,
                use_bias=self.use_bias,
                name=f"conv_{i}",
            )
            for i, (filters, kernel_size) in enumerate(
                zip(self.nb_filters, kernel_sizes)
            )
        ]

        self.bns = (
            [None] * 2
            + [BatchNormalization(name="bn_0")]
            + [None] * 2
            + [BatchNormalization(name="bn_1")]
            + [None] * 2
            + [BatchNormalization(name="bn_2")]
        )

        self.final_conv = Conv1D(
            1,
            1,
            # kernel_constraint=MaxNorm(axis=[0, 1]),
            activation="sigmoid",
            use_bias=self.use_bias,
            name=f"final_conv",
        )
        self.build_weights()
        self.cumulative_gradients = [
            tf.Variable(tf.zeros_like(this_var), trainable=False)
            for this_var in self.trainable_variables
        ]
        self.cumulative_counter = tf.Variable(tf.constant(0), trainable=False)

    # ----------------------------------------------------------------------
    def build_weights(self):
        """ Explicitly build the weights."""
        input_shapes = [self.f_dims] + self.nb_filters[:-1]
        for conv, input_shape in zip(self.convs, input_shapes):
            conv.build((input_shape,))

        self.final_conv.build((self.nb_filters[-1],))

        for bn, input_shape in zip(self.bns, self.nb_filters):
            if bn is not None:
                bn.build((None, None, None, None, input_shape))

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
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x)
            if bn is not None:
                x = bn(x)
        return tf.squeeze(self.final_conv(x), [-2, -1])

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
            loss = self.compiled_loss(
                y, y_pred, regularization_losses=self.losses
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
