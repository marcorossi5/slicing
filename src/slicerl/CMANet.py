# This file is part of SliceRL by M. Rossi
from numpy.lib.arraysetops import isin
from slicerl.AbstractNet import AbstractNet
from slicerl.layers import Attention, GlobalFeatureExtractor
from slicerl.config import TF_DTYPE
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Concatenate, TimeDistributed, InputLayer
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

        # input layer
        self.input_layer = InputLayer(
            type_spec=tf.RaggedTensorSpec(
                shape=(None, None, self.f_dims), dtype=TF_DTYPE
            ),
            name="pc",
        )

        # attention layers
        self.att_filters = [8, 16, 32, 64, 128]
        ifilters = [self.f_dims] + self.att_filters[:-1]
        self.atts = [
            Attention(i, f, name=f"att_{i}")
            for i, (i, f) in enumerate(zip(ifilters, self.att_filters))
        ]

        # MLPs
        self.fc_filters = [128, 64, 32, 16, 8, 4, 2]
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

        self.gfe = GlobalFeatureExtractor(name="feat_extractor")

        self.final_fc = Dense(1, use_bias=self.use_bias, name=f"fc_final")
        # self.build_weights()

    # ----------------------------------------------------------------------
    def build_weights(self):
        """ Explicitly build the weights."""
        input_shapes = [self.f_dims] + self.att_filters[:-1]
        for att, input_shape in zip(self.atts, input_shapes):
            att.build((None, None, input_shape))

        self.gfe.build((None, None, self.att_filters[-1]))

        input_shapes = [self.att_filters[-1] * 3] + self.fc_filters[:-1]
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
        x = self.input_layer(inputs)
        for att in self.atts:
            x = att(x)

        x = self.gfe(x)

        for fc in self.fcs:
            x = fc(x)

        act = tf.squeeze(self.final_fc(x), axis=-1)
        return sigmoid(act)

    # ----------------------------------------------------------------------
    def model(self):
        inputs = Input(shape=(None, self.f_dims), name="input")
        return Model(inputs=inputs, outputs=self.call(inputs), name=self.name)


from tensorflow.keras import Sequential


def get_dummy_net():
    model = Sequential(
        [
            Input(shape=[None, 6], dtype=TF_DTYPE, ragged=True),
            Attention(6, 6, name="att_0"),
            Attention(8, 8, name="att_1"),
            GlobalFeatureExtractor(name="extractor"),
            Dense(4, name="dense_0"),
            Dense(1, name="dense_1"),
        ]
    )
    model.compile(loss="binary_crossentropy", optimizer="rmsprop")
    return model
