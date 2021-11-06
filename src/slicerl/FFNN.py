# This file is part of SliceRL by M. Rossi
from slicerl.AbstractNet import AbstractNet
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Concatenate
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.activations import sigmoid


# ======================================================================
class Head(Layer):
    """ Class defining Spatial Encoding Attention Convolutional Layer. """

    # ----------------------------------------------------------------------
    def __init__(
        self,
        filters,
        nb_head,
        dropout,
        activation="relu",
        name="head",
        **kwargs,
    ):
        """
        Parameters
        ----------
            - dh              : int, number of hidden feature dimensions
            - do              : int, number of output dimensions
            - ds              : int, number of spatial encoding output dimensions
            - K               : int, number of nearest neighbors
            - dims            : int, point cloud spatial dimensions
            - f_dims          : int, point cloud feature dimensions
            - locse_nb_layers : int, number of hidden layers in LocSE block
            - activation      : str, MLP layer activation
            - use_bias        : bool, wether to use bias in MLPs
        """
        super(Head, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.nb_head = nb_head
        self.dropout = dropout
        self.activation = activation
        self.lyrs = []

    # ----------------------------------------------------------------------
    def build(self, input_shape):
        """
        Note: the MLP kernel has size 1. This means that point features are
              not mixed across neighbours.
        """
        for i, filters in enumerate(self.filters[1:]):
            ishape = input_shape if i == 0 else (self.filters[i],)
            l = Dense(
                filters,
                activation=self.activation,
                input_shape=ishape,
                name=f"dense_{self.nb_head}_{i}",
            )
            self.lyrs.append(l)
            if i % 3 == 0:
                l = BatchNormalization(
                    input_shape=(None, filters), name=f"bn_{self.nb_head}_{i}"
                )
                self.lyrs.append(l)
                l = Dropout(
                    self.dropout, input_shape=(filters,), name=f"do_{self.nb_head}_{i}"
                )
                self.lyrs.append(l)

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
        x = inputs
        for layer in self.lyrs:
            x = layer(x)
        return x

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


# ======================================================================
class FFNN(AbstractNet):
    """ Class deifining a simple feed-forward NN. """

    def __init__(
        self,
        batch_size=1,
        f_dims=2,
        activation="relu",
        use_bias=True,
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
        super(FFNN, self).__init__(f_dims=f_dims, name=name, **kwargs)

        # store args
        self.batch_size = int(batch_size)
        self.f_dims = f_dims
        self.activation = activation
        self.use_bias = use_bias
        self.dropout = 0.2

        # store some useful parameters
        self.filters = [
            self.f_dims,
            54,
            50,
            46,
            42,
            36,
            32,
            28,
            24,
            20,
            16,
            12,
            8,
            4,
            2,
            1,
        ]

        self.nb_heads = 1
        self.heads = []
        for ih in range(self.nb_heads):
            self.heads.append(
                Head(
                    self.filters,
                    ih,
                    self.dropout,
                    activation=self.activation,
                    name=f"head_{ih}",
                )
            )

        self.concat = Concatenate(axis=-1, name="cat")
        # self.final_dense = Dense(1, activation=sigmoid, name="final")
        # self.final_dense.build(input_shape=(len(self.heads) * self.filters[-1],))

    # ----------------------------------------------------------------------
    def call(self, inputs):
        """
        Parameters
        ----------
            - inputs : tf.Tensor, of shape=(B, nb_feats)

        Returns
        -------
            tf.Tensor, output tensor of shape=(B,N,nb_classes)
        """
        results = []
        for head in self.heads:
            results.append(head(inputs))

        return sigmoid(results[0])
        return self.final_dense(self.concat(results))

    # ----------------------------------------------------------------------
    def model(self):
        inputs = Input(shape=(self.f_dims), name="pc")
        return Model(inputs=inputs, outputs=self.call(inputs), name=self.name)
