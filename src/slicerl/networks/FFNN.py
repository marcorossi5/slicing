# This file is part of SliceRL by M. Rossi
from slicerl.networks.AbstractNet import AbstractNet
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Concatenate
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.activations import sigmoid


# ======================================================================
class Head(Model):
    """Class defining Spatial Encoding Attention Convolutional Layer."""

    # ----------------------------------------------------------------------
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


# ======================================================================
class FFNN(AbstractNet):
    """Class deifining a simple feed-forward NN."""

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
            128,
            256,
            128,
            64,
            32,
            16,
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
        self.final_dense = Dense(1, activation=sigmoid, name="final")
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

        return self.final_dense(self.concat(results))

    # ----------------------------------------------------------------------
    def model(self):
        inputs = Input(shape=(self.f_dims), name="pc")
        return Model(inputs=inputs, outputs=self.call(inputs), name=self.name)
