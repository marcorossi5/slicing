# This file is part of SliceRL by M. Rossi
from slicerl.RandLANet import DilatedResBlock, Predictions
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Conv1D,
)
from tensorflow.keras.constraints import MaxNorm


#======================================================================
class DRBNet(Model):
    """ Class deifining DRB-Net. """
    def __init__(self, dims=2, f_dims=2, nb_classes=128, K=16, scale_factor=2,
                 nb_layers=4, activation='relu', use_bias=True, fc_type='conv',
                 dropout=0.1, net_type='DRB', name='DRB-Net', **kwargs):
        """
        Parameters
        ----------
            - dims         : int, point cloud spatial dimensions
            - f_dims       : int, point cloud feature dimensions
            - nb_classes   : int, number of output classes for each point in cloud
            - K            : int, number of nearest neighbours to find
            - scale_factor : int, scale factor for down/up-sampling
            - nb_layers    : int, number of inner enocding/decoding layers
            - fc_type      : str, either 'conv' or 'dense' for the final FC
                             layers
            - dropout      : float, dropout percentage in final FC layers
        """
        super(DRBNet, self).__init__(name=name, **kwargs)

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
        self.fc_units = 16
        self.drbs_units = [32, 48, 64, 96, 128]
        self.drbs_iunits = [self.fc_units] + self.drbs_units[:-1]

        # build layers
        if self.fc_type == 'conv':
            self.fc = Conv1D(self.fc_units, 16,padding='same',
                             input_shape=(None,None,None,self.f_dims),
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
                all_cached=True,
                name=f'DRB{i}'
                           ) \
                                for i, (iunits, units) in enumerate(zip(self.drbs_iunits, self.drbs_units))
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

        feats = self.fc(feats)
        residuals.append(feats)

        feats = self.drbs[0]( [pc, feats] )
        cache = self.drbs[0].locse_0.cache

        for drb in self.drbs:
            drb.locse_0.cache = cache
            feats = drb( [pc, feats] )

        return feats # logits

    #----------------------------------------------------------------------
    def model(self):
        pc = Input(shape=(None, self.dims), name='pc')
        feats = Input(shape=(None, self.f_dims), name='feats')
        return Model(inputs=[pc, feats], outputs=self.call([pc,feats]), name=self.name)

    #----------------------------------------------------------------------
    def get_prediction(self, inputs):
        """
        Predict over a iterable of inputs

        Parameters
        ----------
            - inputs : list, elements of shape [(1,N,dims), (1,N,f_dims)]

        Returns
        -------
            Predictions object
        """
        predictions = [ tf.squeeze(self.predict_on_batch(inp), 0) \
                            for inp in inputs]
        return Predictions(predictions)
