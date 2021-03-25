# This file is part of SliceRL by M. Rossi

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
        self.actor = actor

    #----------------------------------------------------------------------
    def _slicing(self, event):
        """Apply subtractor to an event."""
        state=tree.state()
        
        batch = self.process_state_batch([state])
        action = self.actor.predict_on_batch(batch).flatten()
