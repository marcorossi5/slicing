# This file is part of SliceRL by M. Rossi
from slicerl.AbstractNet import AbstractNet
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras import Input, Model

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
        self.input_shapes = [
            self.f_dims,
            self.f_dims * 2,
            self.f_dims * 4,
            self.f_dims * 8,
            128,
        ]

        self.fcs = []
        for i, filters in enumerate(self.input_shapes[1:]):
            l = Dense(filters, activation=self.activation, name=f"dense_{i}")
            l.build(input_shape=(self.input_shapes[i],))
            self.fcs.append(l)

            l = BatchNormalization(name=f"bn_{i}")
            l.build(input_shape=(None, filters))
            self.fcs.append(l)

            l = Dropout(self.dropout, name=f"do_{i}")
            l.build(input_shape=(filters,))
            self.fcs.append(l)

        self.final_dense = Dense(1, activation="sigmoid", name="final")
        self.final_dense.build(input_shape=(self.input_shapes[-1],))

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
        inputs
        for layer in self.fcs:
            inputs = layer(inputs)
        return self.final_dense(inputs)

    # ----------------------------------------------------------------------
    def model(self):
        inputs = Input(shape=(self.f_dims), name="pc")
        return Model(inputs=inputs, outputs=self.call(inputs), name=self.name)
