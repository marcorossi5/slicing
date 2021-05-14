# This file is part of SliceRL by M. Rossi
from slicerl.layers import AbstractNet, DilatedResBlock, RandomSample, UpSample

import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Conv1D,
)
from tensorflow.keras.constraints import MaxNorm



#======================================================================
class RandLANet(AbstractNet):
    """ Class deifining RandLA-Net. """
    def __init__(self, dims=2, f_dims=2, nb_classes=128, K=16, scale_factor=2,
                 nb_layers=4, activation='relu', use_bias=True, fc_type='conv',
                 dropout=0.1, net_type='RandLA', name='RandLA-Net', **kwargs):
        """
        Parameters
        ----------
            - nb_classes   : int, number of output classes for each point in cloud
            - K            : int, number of nearest neighbours to find
            - scale_factor : int, scale factor for down/up-sampling
            - nb_layers    : int, number of inner enocding/decoding layers
            - fc_type      : str, either 'conv' or 'dense' for the final FC
                             layers
            - dropout      : float, dropout percentage in final FC layers
        """
        super(RandLANet, self).__init__(name=name, **kwargs)

        # store args
        self.dims         = dims
        self.f_dims       = f_dims
        self.nb_classes   = nb_classes
        self.K            = int(K)
        self.scale_factor = scale_factor
        self.nb_layers    = nb_layers
        self.activation   = activation
        self.use_bias     = use_bias
        self.fc_type      = fc_type
        self.dropout_perc = dropout

        # store some useful parameters
        self.fc_units = [32, 64, 128, self.nb_classes]
        self.fc_acts  = [self.activation]*3 + ['linear']
        self.latent_f_dim = 32
        self.enc_units  = [
            self.latent_f_dim*2**i \
                for i in range(self.nb_layers)
                          ]
        self.enc_iunits = [8] + self.enc_units[:-1]
        self.dec_units  = self.enc_units[-2::-1] + [self.fc_units[0]]
        self.fc_iunits  = self.dec_units[-1:] + self.fc_units[:-1]

        # build layers
        if self.fc_type == 'conv':
            self.fcs = [
                Conv1D(units, 16, padding='same', input_shape=(None,None,None,iunits),
                              # kernel_regularizer='l2',
                              # bias_regularizer='l2',
                              kernel_constraint=MaxNorm(axis=[0,1]),
                              activation=act, name=f'fc{i+1}') \
                    for i, (iunits, units, act) in enumerate(zip(self.fc_iunits[1:], self.fc_units[1:], self.fc_acts[1:]))
            ]
            self.fcs.insert(0, Dense(self.fc_units[0],
                                     # kernel_regularizer='l2',
                                     # bias_regularizer='l2',
                                     activation=self.fc_acts[0], name=f'fc0') )
        elif self.fc_type == 'dense':
            self.fcs = [
                Dense(units,
                      # kernel_regularizer='l2',
                      # bias_regularizer='l2',
                      kernel_constraint=MaxNorm(), activation=act, name=f'fc{i}') \
                    for i, (units, act) in enumerate(zip(self.fc_units, self.fc_acts))
            ]
        else:
            raise NotImplementedError(f"First and final layers must be of type 'conv'|'dense', not {self.fc_type}")

        if self.dropout_perc:
            self.dropout = [Dropout(self.dropout_perc, name=f'dropout{i+1}') \
                        for i in range(len(self.fc_units[1:-1]))]
        else:
            self.dropout = [[] for i in range(len(self.fc_units[1:-1]))]

        self.encoding   = self.build_encoder()
        self.middle_MLP = Conv1D(self.enc_units[-1], 1,
                              activation=self.activation,
                              use_bias=self.use_bias,
                              # kernel_regularizer='l2',
                              # bias_regularizer='l2',
                              name='MLP')
        self.decoding   = self.build_decoder()

    #----------------------------------------------------------------------
    def build_encoder(self):
        self.encoder = [
            DilatedResBlock(
                input_units=iunits, units=units, K=self.K,
                activation=self.activation, use_bias=self.use_bias,
                name=f'DRB{i}'
                           ) \
                                for i, (iunits, units) in enumerate(zip(self.enc_iunits, self.enc_units))
                       ]
        self.RSs = [
            RandomSample(self.scale_factor, name=f'RS{i}') \
                for i in range(self.nb_layers)
                   ]

    #----------------------------------------------------------------------
    def build_decoder(self):
        self.dec_iunits = self.enc_units[::-1]

        self.USs = [
            UpSample(
                iunits, units, self.scale_factor, activation=self.activation,
                use_bias=self.use_bias, name=f'US{i}'
                    ) \
                for i, (iunits, units) in enumerate(zip(self.dec_iunits, self.dec_units))
                   ]

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

        residuals = []
        n_idxs    = []
        rndss     = []

        feats = self.fcs[0](feats)
        residuals.append(feats)

        for drb, rs  in zip(self.encoder, self.RSs):
            feats = drb( [pc, feats] )
            pc, feats, n_idx, rnds = rs([pc, feats])
            n_idxs.append(n_idx)
            rndss.append(rnds)
            residuals.append(feats)

        feats = self.middle_MLP(feats)
        for i, (us, interp, ups, res) in enumerate(zip(self.USs, n_idxs[::-1], rndss[::-1], residuals[-2::-1])):
            feats = us([feats, interp, ups, res])

        for fc, do in zip(self.fcs[1:-1], self.dropout):
            feats = fc(feats)
            if self.dropout_perc:
                feats = do(feats)

        return self.fcs[-1](feats) # logits

