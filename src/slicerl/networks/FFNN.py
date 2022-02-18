# This file is part of SliceRL by M. Rossi
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras import Input, Model
from tensorflow.keras.activations import sigmoid
from .AbstractNet import AbstractNet
from .layers import Head


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
