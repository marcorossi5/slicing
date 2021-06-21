# This file is part of SliceRL by M. Rossi
from slicerl.layers import AbstractNet, SEAC

import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Reshape
from tensorflow.keras.constraints import MaxNorm

# ======================================================================
class SeacNet(AbstractNet):
    """ Class deifining SEAC-Net. """

    def __init__(
        self,
        dims=2,
        f_dims=2,
        K=16,
        locse_nb_layers=3,
        activation="relu",
        use_bias=True,
        name="SEAC-Net",
        **kwargs,
    ):
        """
        Parameters
        ----------
            - dims            : int, point cloud spatial dimensions
            - f_dims          : int, point cloud feature dimensions
            - K               : int, number of nearest neighbours to find
            - locse_nb_layers : int, number of hidden layers in LocSE block
            - use_bias        : bool, wether to use bias or not
        """
        super(SeacNet, self).__init__(name=name, **kwargs)

        # store args
        self.dims = dims
        self.input_dims = self.dims + 2
        self.f_dims = f_dims
        self.K = int(K)
        self.locse_nb_layers = locse_nb_layers
        self.activation = activation
        self.use_bias = use_bias

        # store some useful parameters
        ds = [self.f_dims, 5, 7, 10, 15, 1]
        self.seacs = [
            SEAC(
                dh=dh,
                do=do,
                K=self.K,
                locse_nb_layers=self.locse_nb_layers,
                activation=self.activation,
                use_bias=self.use_bias,
                f_dims=self.f_dims,
                name=f"seac{i+1}",
            )
            for i, (dh, do) in enumerate(zip(ds[1:-1], ds[2:]))
        ]
        self.seacs.insert(
            0,
            SEAC(
                dh=ds[0],
                do=ds[1],
                K=self.K,
                locse_nb_layers=self.locse_nb_layers,
                use_cache=False,
                use_bias=self.use_bias,
                activation=self.activation,
                f_dims=self.f_dims,
                name="seac0",
            ),
        )
        self.final_conv = Conv1D(
            1 + self.K,
            1,
            input_shape=(None, 1 + self.K),
            kernel_constraint=MaxNorm(axis=[0, 1]),
            activation="sigmoid",
            use_bias=self.use_bias,
            name=f"final_conv",
        )
        self.reshape = Reshape((-1, 1 + self.K), name="reshape")

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
        pc, feats = inputs
        edges = self.seacs[0]([pc, feats])
        cache = self.seacs[0].locse.cache

        for seac in self.seacs[1:]:
            seac.locse.cache = cache
            edges = seac([pc, edges])

        edges = self.reshape(edges + feats)

        return self.final_conv(edges)
