# This file is part of SliceRL by M. Rossi
from slicerl.layers import AbstractNet, SEAC

import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Reshape
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.activations import sigmoid

# ======================================================================
class CMNet(AbstractNet):
    """ Class deifining CM-Net: Cluster merging network. """

    def __init__(
        self,
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
        self.f_dims = f_dims
        self.activation = activation
        self.use_bias = use_bias

        # store some useful parameters
        nb_filters = [32, 64, 32, 8, 16, 8]
        kernel_sizes = [2] + [1] * (len(nb_filters) - 1)
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
                zip(nb_filters, kernel_sizes)
            )
        ]

        self.final_conv = Conv1D(
            1,
            1,
            # kernel_constraint=MaxNorm(axis=[0, 1]),
            activation="sigmoid",
            use_bias=self.use_bias,
            name=f"final_conv",
        )

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
        for conv in self.convs:
            x = conv(x)
        return tf.squeeze(self.final_conv(x), [-2, -1])
