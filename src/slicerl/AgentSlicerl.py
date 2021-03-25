# This file is part of SliceRL by M. Rossi

from slicerl.Slicer import ContinuousSlicer
from rl.agents.ddpg import DDPGAgent
from tensorflow.keras.models import model_from_json

#======================================================================
class DDPGAgentSlicerl(DDPGAgent):
    """DDPG Agent for pile up subtraction"""

    #----------------------------------------------------------------------
    def __init__(self, *args, **kwargs):
        """Initialize the DQN agent."""
        super(DDPGAgentSlicerl, self).__init__(*args, **kwargs)

    #----------------------------------------------------------------------
    def save(self, filepath, overwrite=False, include_optimizer=True):
        """Save the model to file."""
        self.model.save(filepath, overwrite=overwrite,
                        include_optimizer=include_optimizer)

    #----------------------------------------------------------------------
    def load_model(self, filepath, custom_objects=None, compile=True):
        """Load model from file"""
        self.model = load_model(filepath, custom_objects=custom_objects,
                                compile=compile)
        self.update_target_model_hard()

    #----------------------------------------------------------------------
    def load_with_json(self, jsonfile, weightfile):
        """
        Load model from a json file with the architecture, and an h5 file with weights.
        """
        # read architecture card
        with open(jsonfile) as f:
            arch = json.load(f)
        self.model = model_from_json(arch)
        self.model.load_weights(weightfile)
        self.update_target_model_hard()

    #----------------------------------------------------------------------
    def slicer(self):
        """Return the current subtractor."""
        return ContinuousSlicer(self.actor, self.test_policy)
        
    
