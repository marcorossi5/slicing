# This file is part of SliceRL by M. Rossi

import os
import numpy as np
import math, json
from abc import ABC, abstractmethod
from rl.policy import GreedyQPolicy
from tensorflow.keras.models import model_from_json

#======================================================================
class AbstractSlicer(ABC):
    """AbstractSlicer class."""

    #----------------------------------------------------------------------
    def __call__(self, event):
        """Apply the subtractor to an event and returned the cleaned event."""
        # TODO: implement
        return self._slicing(event)

    #----------------------------------------------------------------------
    @abstractmethod
    def _slicing(self, event):
        return event
        
class ContinuousSlicer(AbstractSlicer):
    """ContinuousSlicer class that acts on an event using a discrete action."""

    #---------------------------------------------------------------------- 
    def __init__(self, actor=None):
        """Initialisation of the subtractor."""
        # load a ddpg instance here
        self.actor = actor
        self.action_max = 1.0
        self.eps = np.finfo(np.float32).eps

    #----------------------------------------------------------------------
    def _slicing(self, event):
        """
        Apply subtractor to an event. Just overwrite calohit status with slice
        index, do not keep track of slices cumulative statistics. Delegate
        slice checks to diagnostics.
        """
        slice_count = 0

        for i in range(len(event.calohits)):
            c = event.calohits[i]
            if i == 0:
                # first calohit seeds a new slice automatically
                c.status = 0
                slice_count += 1
                continue

            state = event.state(i)
            action = self.actor.predict_on_batch(np.array([[state]])).flatten()
            action = np.clip(action, -self.action_max, self.action_max-self.eps)

            if action[0] >= 0:
                idx = math.floor(action*slice_count)
                c.status = idx
            else:
                c.status = slice_count
                slice_count += 1

        return event

    #----------------------------------------------------------------------
    def load_with_json(self, jsonfile, weightfile):
        """
        Load model from a json file with the architecture, and an h5 file with weights.
        First actor, second critic.
        """
        # read actor architecture card
        filename, extension = os.path.splitext(jsonfile)
        actor_jsonfile = filename + '_actor' + extension
        with open(actor_jsonfile) as f:
            arch = json.load(f)
        self.actor = model_from_json(arch)

        filename, extension = os.path.splitext(weightfile)
        actor_weightfile = filename + '_actor' + extension
        self.actor.load_weights(actor_weightfile)

    #----------------------------------------------------------------------
    def save(self, filepath, overwrite=False, include_optimizer=True):
        """Save the model to file."""
        self.actor.save(filepath, overwrite=overwrite,
                        include_optimizer=include_optimizer)

    #----------------------------------------------------------------------
    def load_model(self, filepath, custom_objects=None, compile=True):
        """Load model from file"""
        # TODO: check this
        self.actor = load_model(filepath, custom_objects=custom_objects,
                                compile=compile)

    #----------------------------------------------------------------------
    def save_weights(self, filepath, overwrite=False):
        """Save the weights of model to file."""
        self.actor.save_weights(filepath, overwrite=overwrite)

    #----------------------------------------------------------------------
    def load_weights(self, filepath):
        """Load weights of model from file"""
        self.actor.load_weights(filepath)
