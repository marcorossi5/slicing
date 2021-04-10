# This file is part of SliceRL by M. Rossi

import random, math, pprint
from slicerl.read_data import Events
from slicerl.Event import Event
from gym import spaces, Env
from gym.utils import seeding
import numpy as np
from slicerl.tools import dice_loss, mse, m_lin_fit, pearson_distance, efficiency_rejection_rate_loss
from copy import deepcopy

#======================================================================
class SlicerlEnv(Env):
    """Class defining a gym environment for the slicing algorithm."""
    #----------------------------------------------------------------------
    def __init__(self, hps, low, high):
        """
        Initialisation of the environment using a dictionary with hyperparameters
        and a lower and upper bound for the observable state.

        The hps dictionary should have the following entries:
        - fn: filename of data set
        - nev: number of events (-1 for all)
        - reward: type of reward function (cauchy, gaussian, ...)
        """
        self.max_hits = hps['max_hits']
        # read in the events
        reader = Events(hps['fn'], hps['nev'], hps['min_hits'], hps['max_hits'])
        self.events = reader.values()

        # set up action and observation space
        self.action_space      = spaces.Discrete(2)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        
        # set up some internal parameters
        self.seed()
        self.viewer = None

        # set the reward function
        self.width = hps['width']
        if hps['reward']=='cauchy':
            self.__reward=self.__reward_Cauchy
        elif hps['reward']=='gaussian':
            self.__reward=self.__reward_Gaussian
        elif hps['reward']=='exponential':
            self.__reward=self.__reward_Exponential
        elif hps['reward']=='inverse':
            self.__reward=self.__reward_Inverse
        else:
            raise ValueError('Invalid reward: %s'%hps['reward'])

        self.description= '%s with\n%s' % (self.__class__.__name__,pprint.pformat(hps))
        print('Setting up %s' % self.description)

    #----------------------------------------------------------------------
    def reset_current_event(self):
        """Reset the current event."""
        self.event       = deepcopy(random.choice(self.events))
        self.index       = -1
        self.slices      = [[] for i in range(self.nbins)]
        self.set_next_node()

    #----------------------------------------------------------------------
    def set_next_node(self):
        """Set the current calohit using the event list."""
        # print('setting node up')
        while self.index < len(self.event):
            self.index += 1
            if self.index < len(self.event):
                if (self.event.calohits[self.index].status is None)\
                    or (self.event.calohits[self.index].status == -1):
                    break
        self.state = self.event.state(self.index)

    #----------------------------------------------------------------------
    def __reward_Cauchy(self, x):
        """A cauchy reward function."""
        return 1.0/(math.pi*(1.0 + (x*x)))

    #----------------------------------------------------------------------
    def __reward_Gaussian(self, x):
        """A gaussian reward function."""
        return np.exp(-x*x/2.0)

    #----------------------------------------------------------------------
    def __reward_Exponential(self, x):
        """A negative exponential reward function."""
        return  np.exp(-x)

    #----------------------------------------------------------------------
    def __reward_Inverse(self, x):
        """An inverse reward function."""
        return min(1.0, 1.0/(x + 0.5))

    #----------------------------------------------------------------------
    def reward(self, slice_state, mc_state):
        """Full reward function."""
        x = dice_loss(slice_state, mc_state)
        return self.__reward(x/self.width)

    #----------------------------------------------------------------------
    def seed(self, seed=None):
        """Initialize the seed."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    #----------------------------------------------------------------------
    def reset(self):
        """Reset the event to a new randomly selected one."""
        self.reset_current_event()
        return self.state

    #----------------------------------------------------------------------
    def render(self, mode='human'):
        """Add potential intermediary output here"""
        pass

    #----------------------------------------------------------------------
    def close(self):
        if self.viewer: self.viewer.close()

#======================================================================
class SlicerlEnvDiscrete(SlicerlEnv):
    """Class defining a gym environment for the discrete slicing algorithm."""
    #----------------------------------------------------------------------
    def __init__(self, *args, **kwargs):
        super(SlicerlEnvDiscrete, self).__init__(*args, **kwargs)
        self.action_space = spaces.Discrete(self.nbins)
        self.discrete = True

        # set up the slice list
        self.nbins  = 128                             # the maximum number of slices
        self.slices = [[] for i in range(self.nbins)] # contains slice calohit idx

    #----------------------------------------------------------------------
    def step(self, action):
        """
        Perform a step using the current calohit in the event, deciding which
        slice to add it to. Action is the slice index.
        """
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        c = self.event.calohits[self.index]
        mc_state = c.mc_state

        # overwrite calohit status attribute with slice index
        c.status = action

        # add calohit (E,x,z) to cumulative slice state
        self.slices[action].append(self.index)
        m = self.slices[action]
        x = self.event.array[1][m]
        z = self.event.array[2][m]
        slice_state = np.array([
                                self.event.array[0][m].sum(),
                                x.mean(),
                                x.std(),
                                z.mean(),
                                z.std(),
                                len(m),
                                m_lin_fit(x, z),
                                pearson_distance(x, z)
                               ])
        # compute reward        
        reward = self.reward(slice_state, mc_state)

        # move to the next node in the clustering sequence
        self.set_next_node()

        # if we are at the end of the declustering list, then we are done for this event.
        done = bool(self.index >= len(self.event))

        # return the state, reward, and status
        return self.state, reward, done, {}

#======================================================================
class SlicerlEnvContinuous(SlicerlEnv):
    """Class defining a gym environment for the continuous slicing algorithm."""
    #----------------------------------------------------------------------
    def __init__(self, hps, **kwargs):
        super(SlicerlEnvContinuous, self).__init__(hps, **kwargs)
        self.action_space = spaces.Box(low=0., high=1.,
                                    shape=(hps['max_hits'],), dtype=np.float32)
        self.discrete         = False
        self.threshold        = 0.5
        self.nb_max_episode_steps = 128
    
    #----------------------------------------------------------------------
    def set_next_node(self):
        """Prepare the current step to seed a new slice."""
        # print('setting node up')
        self.index += 1
        self.mc_state = (self.event.ordered_mc_idx == self.index)
        self.state = deepcopy(self.event.state())

    #----------------------------------------------------------------------
    def reset_current_event(self):
        """Reset the current event."""
        self.event       = deepcopy(random.choice(self.events))
        self.index       = -1
        self.set_next_node()

    #----------------------------------------------------------------------
    def reward(self, slice_state, mask_state, mc_state):
        """
        Full reward function. slice_state contain actor scores in range [0,1].
        mask_state and mc_state are boolean masks.
        """
        er_rate_loss = efficiency_rejection_rate_loss(mask_state, mc_state)
        dice         = dice_loss(slice_state, mc_state.astype(slice_state.dtype))
        x = er_rate_loss + dice

        return self._SlicerlEnv__reward(x/self.width)

    #----------------------------------------------------------------------
    def step(self, action):
        """
        Perform a step using the current calohit in the event, deciding which
        slice to add it to. Action is a score that falls inside a bin in the
        unit interval. The bin index is the slice index.
        """
        # action shape is (max_hits,)
        # clip action since random process could drift the action out of action_space bounds
        action = np.clip(action, 0., 1.)
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        valid_action   = action[:self.event.num_calohits]
        current_status = self.event.point_cloud[-1][:self.event.num_calohits]

        # threshold the action to output a mask and apply it to the current status
        m = np.logical_and(current_status == -1, valid_action > self.threshold)
        current_status[m] = self.index

        # print(f"action range: [{valid_action.min():.5f}, {valid_action.max():.5f}], update hits: {np.count_nonzero(m)},",
        #       f"to update: {np.count_nonzero(self.mc_state)}, num calohits: {self.event.num_calohits}, new status vector range: [{current_status.min()}, {current_status.max()}]")
        # compute reward only on valid calohits, not padded ones
        # penalize if no calohit is predicted above threshold
        penalty = -2 if np.count_nonzero(m) == 0 else 0
        reward = self.reward(valid_action, m, self.mc_state) + penalty

        # prepare next step
        self.set_next_node()

        # if we are at the end of the declustering list, then we are done for this event.
        done = bool( (np.count_nonzero(current_status == -1) == 0) \
                     or (self.index >= self.nb_max_episode_steps) )

        # return the state, reward, and status
        return self.state, reward, done, {}

"""
TODO: implement a proper reward function (bce/dice_loss)
Maybe dice loss is better since it's bounded, while bce is not. Moreover, it is
not affected by unbalancing, since we want to identify small cluster inside big
set of calohits.
"""