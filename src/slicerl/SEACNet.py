# This file is part of SliceRL by M. Rossi
from slicerl.layers import AbstractNet, SEAC

import tensorflow as tf
from tensorflow.keras.layers import (
    Conv1D,
    Reshape
)
from tensorflow.keras.constraints import MaxNorm

#======================================================================
class SeacNet(AbstractNet):
    """ Class deifining SEAC-Net. """
    def __init__(self, dims=2, f_dims=2, K=16,
                 nb_final_convs=1, activation='relu',
                 use_bias=True,
                 name='SEAC-Net', **kwargs):
        """
        Parameters
        ----------
            - dims           : int, point cloud spatial dimensions
            - f_dims         : int, point cloud feature dimensions
            - nb_classes     : int, number of output classes for each point in cloud
            - K              : int, number of nearest neighbours to find
            - scale_factor   : int, scale factor for down/up-sampling
            - nv final_convs : int, number of final convolutions
            - fc_type        : str, either 'conv' or 'dense' for the final FC
                               layers
            - dropout        : float, dropout percentage in final FC layers
        """
        super(SeacNet, self).__init__(name=name, **kwargs)

        # store args
        self.dims           = dims
        self.f_dims         = f_dims
        self.K              = int(K)
        self.nb_final_convs = nb_final_convs
        self.activation     = activation
        self.use_bias       = use_bias

        # store some useful parameters
        ds = [2,5,10,5,3,1]
        self.seacs = [
            SEAC(dh=dh, do=do, K=self.K, activation=self.activation, name=f'seac{i+1}') \
                for i,(dh,do) in enumerate(zip(ds[1:-1], ds[2:]))
                     ]
        self.seacs.insert(0,
                          SEAC(dh=ds[0], do=ds[1], K=self.K,
                               use_cache=False, activation=self.activation,
                               name='seac0')
                         )
        self.final_conv = Conv1D(
               1+self.K, 1,
               input_shape=(None,1+self.K),
               kernel_constraint=MaxNorm(axis=[0,1]),
               activation='sigmoid',
               use_bias=self.use_bias,
               name=f'final_conv'
                  )
        self.reshape = Reshape((-1, 1+self.K), name='reshape')
 
    #----------------------------------------------------------------------
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
        edges = self.seacs[0]([pc,feats])
        cache = self.seacs[0].locse.cache

        for seac in self.seacs[1:]:
            seac.locse.cache = cache
            edges = seac([pc,edges])
        
        return self.final_conv( self.reshape(edges) )
