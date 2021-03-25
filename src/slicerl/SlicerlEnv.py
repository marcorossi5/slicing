# This file is part of SliceRL by M. Rossi

import random, math, pprint
from slicerl.read_data import Events
from slicerl.Event import Event
from gym import spaces, Env
from gym.utils import seeding
import numpy as np
from slicerl.tools import quality_metric


class TupleSpace(spaces.Tuple):
    def __init__(self, *args, **kwargs):
        super(TupleSpace,self).__init__(*args, **kwargs)
        # watch out ! this breaks if some space has shape of type NoneType
        print(list(space.shape for space in self.spaces))
        self.shape = tuple([sum(space.shape for space in self.spaces)])

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
        # read in the events
        reader = Events(hps['fn'], hps['nev'], hps['min_hits'])
        self.events = reader.values()

        # set up the mass parameters and initial state
        self.root          = None

        # set up observation and action space
        self.action_space      = spaces.Discrete(2)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        # self.observation_space = TupleSpace(
        #                                 (spaces.Box(flow, fhigh, dtype=np.float32), \
        #                                 spaces.Discrete(ihigh) )
        #                                    )
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
        self.event = random.choice(self.events)
        self.index = -1
        self.set_next_node()

    #----------------------------------------------------------------------
    def set_next_node(self):
        """Set the current particle using the event list."""
        # print('setting node up')
        while True:
            self.index += 1
            if (len(self.event.calohits) >= self.index)\
               or (self.event.calohits[self.index].status is None):
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
    def reward(self, ipair):
        """Full reward function."""
        # TODO:
        # implement reward by comparing properties of paired jets
        j_noPU,j = self.event.jet_pairs[ipair]
        x = quality_metric(j_noPU, j)
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
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)


    #----------------------------------------------------------------------
    def step(self, action):
        """
        Perform a step using the current calohit in the event, deciding which
        slice to add it to.
        """
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        p = self.event.particles[self.index]
        ipair = p.index
        neighbours = p.neighbours

        scalefact = action[0]
        # rescale particle by scalefact, update corresponding jet
        j = self.event.jet_pairs[ipair][1]
        r = 1.0 - scalefact
        j.reset(j.px - r*p.px, j.py - r*p.py, j.pz - r*p.pz, j.E - r*p.E)
        p.reset(scalefact*p.px, scalefact*p.py, scalefact*p.pz, scalefact*p.E)

        # calculate a reward
        reward = self.reward(ipair)

        # move to the next node in the clustering sequence
        self.set_next_node()

        # if we are at the end of the declustering list, then we are done for this event.
        done = bool(len(self.event) <= self.index)

        # return the state, reward, and status
        return self.state, reward, done, {}

"""
TODO: split the observation space in 3-dim Box plut
TODO: implement the env.step function
"""