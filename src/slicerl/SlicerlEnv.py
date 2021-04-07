# This file is part of SliceRL by M. Rossi

import random, math, pprint
from slicerl.read_data import Events
from slicerl.Event import Event
from gym import spaces, Env
from gym.utils import seeding
import numpy as np
from slicerl.tools import quality_metric
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
        # add small penalty for not clustering
        self.penalty = hps['penalty']

        # read in the events
        self.k = hps['k']
        reader = Events(hps['fn'], hps['nev'], k=hps['k'], min_hits=hps['min_hits'])
        self.events = reader.values()

        # set up the count parameters and initial state
        self.slice_count   = 0
        self.slices        = []

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
        """Reset the current event and insert the first calohit."""
        self.event       = deepcopy(random.choice(self.events))
        self.index       = -1
        self.slice_count = 0
        self.slices      = []
        self.set_next_node()

        # overwrite calohit status with slice index

        c = self.event.calohits[self.index]
        c.status = self.slice_count

        # add calohit (E,x,z) to cumulative slice state
        calohit_state = self.state[0,1:4]
        self.slices.append( calohit_state )
        self.slice_count += 1

        self.set_next_node()

    #----------------------------------------------------------------------
    def set_next_node(self, is_reset=False):
        """Set the current particle using the event list."""
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
        x = quality_metric(slice_state, mc_state)
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
class SlicerlEnvContinuous(SlicerlEnv):
    """Class defining a gym environment for the continuous slicing algorithm."""
    #----------------------------------------------------------------------
    def __init__(self, *args, **kwargs):
        super(SlicerlEnvContinuous, self).__init__(*args, **kwargs)
        self.eps = np.finfo(np.float32).eps
        self.action_max = 1.0
        self.action_space = spaces.Box(low=-self.action_max, high=self.action_max,
                                       shape=(1,), dtype=np.float32)

    #----------------------------------------------------------------------
    def step(self, action):
        """
        Perform a step using the current calohit in the event, deciding which
        slice to add it to.
        """
        action = np.clip(action, -self.action_max, self.action_max-self.eps)
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        c = self.event.calohits[self.index]
        mc_state = c.mc_state
        calohit_state = self.state[0,1:4]

        aggregate = action[0] >= 0
        if aggregate:
            # select the correct existing slice to put the calohit in
            idx = math.floor(action[0]*self.slice_count)

            # overwrite calohit status attribute with slice index
            c.status = idx

            # add calohit (E,x,z) to cumulative slice state
            self.slices[idx] += calohit_state
            slice_state = self.slices[idx]
        else:
            # form a new slice
            # overwrite calohit status with slice index
            c.status = self.slice_count

            # add calohit (E,x,z) to cumulative slice state
            self.slices.append( calohit_state )
            slice_state = calohit_state
            self.slice_count += 1

        # calculate a reward        
        # penalize wrongly seeding new slice and when aggregating to small slice
        large_cluster = np.count_nonzero(c.mc_neighbours_m) >= 5
        penalty = self.penalty / 1e3 * (not aggregate)
        extra_reward =  self.penalty / 1e3 * (large_cluster == aggregate)
        extra_penalty =  self.penalty / 5e1 * (large_cluster and not aggregate)
        # penalize when aggregating if neighbours < 5
        # penalize when seeding new slide if neighbours > 5
        reward = self.reward(slice_state, mc_state) + extra_reward - penalty - extra_penalty

        # move to the next node in the clustering sequence
        self.set_next_node()

        # if we are at the end of the declustering list, then we are done for this event.
        done = bool(self.index >= len(self.event))

        # return the state, reward, and status
        return self.state, reward, done, {}

"""
TODO: split the observation space in 3-dim Box plut
TODO: implement the env.step function
"""