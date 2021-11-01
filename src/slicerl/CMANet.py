# This file is part of SliceRL by M. Rossi
from numpy.lib.arraysetops import isin
from slicerl.AbstractNet import AbstractNet
from slicerl.layers import Attention
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Concatenate, Input
from tensorflow.keras.activations import sigmoid

# ======================================================================
class CMANet(AbstractNet):
    """ Class deifining Cluster Merging with Attention Network. """
    def __init__(
        self,
        batch_size=32,
        f_dims=6,
        activation="relu",
        use_bias=True,
        name="CMANet",
        **kwargs,
    ):
        """
        Parameters
        ----------
            - batch_size: int, batch size
            - f_dims: int, point cloud feature dimensions
            - activation: str, default layer activation
            - use_bias: bool, wether to use bias or not
        """
        super(CMANet, self).__init__(f_dims=f_dims, name=name, **kwargs)

        # store args
        self.batch_size = int(batch_size)
        self.f_dims = f_dims
        self.activation = activation
        self.use_bias = use_bias

        # self.cumulative_gradients = None
        self.cumulative_counter = tf.constant(0)

        # attention layers
        self.att_filters = [8, 16, 32]
        self.atts = [Attention(f,f, name=f"att_{i}") for i, f in enumerate(self.att_filters)]

        # MLPs
        self.fc_filters = [8, 4, 2]
        self.fcs = [
            Dense(
                f,
                # kernel_constraint=MaxNorm(axis=[0, 1]),
                activation=self.activation,
                use_bias=self.use_bias,
                name=f"fc_{i}",
            )
            for i, f in enumerate(self.fc_filters)
        ]

        self.cat = Concatenate(axis=-1, name='cat')

        self.final_fc = Dense(1,
                use_bias=self.use_bias,
                name=f"fc_final"
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
        input_shapes = [self.f_dims] + self.att_filters[:-1]
        for att, input_shape in zip(self.atts, input_shapes):
            att.build((None, None, input_shape))

        input_shapes = [self.att_filters[-1]*3] + self.fc_filters[:-1]
        for fc, input_shape in zip(self.fcs, input_shapes):
            fc.build((None, input_shape))
        
        self.final_fc.build((self.fc_filters[-1],))

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
        for att in self.atts:
            inputs = att(inputs)

        global_max = tf.math.reduce_max(inputs, axis=-2, name='max')
        global_min = tf.math.reduce_min(inputs, axis=-2, name='min')
        global_avg = tf.math.reduce_mean(inputs, axis=-2, name='avg')
        # global_std = tf.math.reduce_std(inputs, axis=-2, name='std') raises NaN
        x = self.cat([global_max, global_min, global_avg])

        for fc in self.fcs:
            x = fc(x)
        
        act = tf.squeeze(self.final_fc(x), axis=-1)
        return sigmoid(act)

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
    
    # ----------------------------------------------------------------------
    def model(self):
        inputs = Input(shape=(None, self.f_dims), name="pc")
        return Model(inputs=inputs, outputs=self.call(inputs), name=self.name)

    # ----------------------------------------------------------------------
    def predict(self, x, batch_size, **kwargs):
        """
        Overrides the predict method of mother class. CMANet accepts only
        tensors of shape (1,N,C) as inputs. This method allows to pass a numpy array of
        tensors
        """
        from tqdm import tqdm
        if isinstance(x, np.ndarray):
            if len(x.shape) == 3:
                return super(CMANet, self).predict(x, batch_size, **kwargs)
            elif len(x.shape) == 1:
                out = []
                for i in tqdm(x):
                    inputs = np.expand_dims(i, 0)
                    out.append(super(CMANet, self).predict(inputs, batch_size, **kwargs))
                print(out, type(out))
                exit()
                return out


