# This file is part of SliceRL by M. Rossi
from slicerl.AbstractNet import AbstractNet
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Concatenate
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
            self.f_dims * 3,
            self.f_dims * 4,
            self.f_dims * 5,
            self.f_dims * 6,
            self.f_dims * 7,
            self.f_dims * 8,
            self.f_dims * 7,
            self.f_dims * 6,
            self.f_dims * 5,
            self.f_dims * 4,
            self.f_dims * 3,
            self.f_dims * 2,
            128,
            1
        ]

        self.heads = []
        for head in self.heads:
            for i, filters in enumerate(self.input_shapes[1:]):
                l = Dense(filters, activation=self.activation, name=f"dense_{i}")
                l.build(input_shape=(self.input_shapes[i],))
                head.append(l)
                if i%3 == 0:
                    l = BatchNormalization(name=f"bn_{i}")
                    l.build(input_shape=(None, filters))
                    head.append(l)
                    l = Dropout(self.dropout, name=f"do_{i}")
                    l.build(input_shape=(filters,))
                    head.append(l)

        self.concat = Concatenate(axis=-1, name='cat')
        self.final_dense = Dense(1, activation="sigmoid", name="final")
        self.final_dense.build(input_shape=(len(self.heads)*self.input_shapes[-1],))

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
            x = inputs
            for layer in head:
                x = layer(x)
            results.append(x)
        
        return self.final_dense(self.concat(results))

    # ----------------------------------------------------------------------
    def model(self):
        inputs = Input(shape=(self.f_dims), name="pc")
        return Model(inputs=inputs, outputs=self.call(inputs), name=self.name)
