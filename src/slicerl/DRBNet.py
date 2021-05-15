# This file is part of SliceRL by M. Rossi
from slicerl.layers import AbstractNet, DilatedResBlock

import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Conv1D,
    Concatenate,
    Dropout
)
from tensorflow.keras.constraints import MaxNorm

#======================================================================
class DRBNet(AbstractNet):
    """ Class deifining DRB-Net. """
    def __init__(self, dims=2, f_dims=2, nb_classes=128, K=16,
                 nb_final_convs=3, activation='relu', use_bias=True, fc_type='conv',
                 dropout=0.1, use_ggf=False, name='DRB-Net', **kwargs):
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
        super(DRBNet, self).__init__(name=name, **kwargs)

        # store args
        self.dims           = dims
        self.f_dims         = f_dims
        self.nb_classes     = nb_classes
        self.K              = int(K)
        self.nb_final_convs = nb_final_convs
        self.activation     = activation
        self.use_bias       = use_bias
        self.fc_type        = fc_type
        self.use_ggf        = use_ggf
        self.dropout_perc   = dropout

        # store some useful parameters
        self.fc_units    = 16
        self.drbs_units  = [32, 48, 64, 96, 128]
        self.caches      = [False] + [True] * len(self.drbs_units[1:])
        self.drbs_iunits = [self.fc_units] + self.drbs_units[:-1]

        # build layers
        self.cat = Concatenate(axis=-1, name='cat')

        if self.fc_type == 'conv':
            self.fc = Conv1D(self.fc_units, 16, padding='same',
                             input_shape=(None,None,None,self.dims + self.f_dims),
                             # kernel_regularizer='l2',
                             # bias_regularizer='l2',
                             kernel_constraint=MaxNorm(axis=[0,1]),
                             activation=self.activation,
                             name=f'fc')
        elif self.fc_type == 'dense':
            self.fc =  Dense(self.fc_units,
                             # kernel_regularizer='l2',
                             # bias_regularizer='l2',
                             kernel_constraint=MaxNorm(),
                             activation=self.activation,
                             name=f'fc')
        else:
            raise NotImplementedError(f"First and final layers must be of type 'conv'|'dense', not {self.fc_type}")

        self.drbs = [
            DilatedResBlock(
                input_units=iunits,
                units=units,
                K=self.K,
                activation=self.activation,
                use_bias=self.use_bias,
                use_ggf=self.use_ggf,
                all_cached=caches,
                name=f'DRMB{i}'
                           ) \
                                for i, (iunits, units, caches) in enumerate(zip(self.drbs_iunits, self.drbs_units, self.caches))
                       ]

        self.final_convs = [
            Conv1D(
                1, 1,
                input_shape=(None,None,self.nb_classes,2),
                # kernel_regularizer='l2',
                # bias_regularizer='l2',
                kernel_constraint=MaxNorm(axis=[0,1]),
                activation='linear',
                use_bias=self.use_bias,
                name=f'final_fc{i}') \
                       for i in range(self.nb_final_convs)
                           ]

        if self.dropout_perc:
            self.dropout = [Dropout(self.dropout_perc, name=f'dropout{i+1}') \
                        for i in range(self.nb_final_convs)]
        else:
            self.dropout = [[] for i in range(self.nb_final_convs)]
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
        N = tf.shape(inputs)[-2]

        cat_feats = self.cat([pc,feats])        

        residuals = []

        feats = self.fc(cat_feats)
        residuals.append(feats)

        feats = self.drbs[0]( [pc, feats] )
        cache = self.drbs[0].locse_0.cache

        for drb in self.drbs[1:]:
            drb.locse_0.cache = cache
            feats = drb( [pc, feats] )
        
        for fc, do in zip(self.final_convs, self.dropout):
            # compute slice size
            mean = tf.reduce_mean(feats, axis=1, keepdims=True)
            size = tf.repeat(
                    mean,
                    N,
                    axis=1)
            # concat size to feats
            feats = tf.stack([feats, size], axis=-1)
            feats = tf.squeeze( fc(feats), axis=-1 )
            if self.dropout_perc:
                feats = do(feats)

        return feats # logits
