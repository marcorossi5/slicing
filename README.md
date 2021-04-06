# SlicingRL

Slicing Problem with Reinforcement Learning for DUNE reconstruction.

## Python Environment

SliceRL requires the following python packages:

- python3 (version 3.8)
- numpy
- gym
- matplotlib
- tensorflow (version 2.3)
- [keras-rl2](https://github.com/wau/keras-rl2.git)
- [energyflow](https://github.com/pkomiske/EnergyFlow.git)
- json
- gzip
- argparse
- [hyperopt](https://github.com/hyperopt/hyperopt.git) (optional)

## Problem

Slicing algorithm clusters calorimeter hits (CaloHits) based on the main primary interacting particle. This repository implements a DDPG agent that learns to recursively put CaloHits into their correct slice. The agent tries to predict the slice index for each CaloHit.

## Environment

The observation space is a three dimensional box, representing input particle (E, x, U) coordinates in the LArTPC. The environment contains a continuous action space in the interval [-1,1].  

The initial state is given by a single hit filling one slice. After that, each hit in an Event is observed: the agent predicts the best action-value that will decide slice index. Action space is divided in the following sectors:

- negative values imply the hit to seed a new cluster. Therefore the number of slices `l` are incremented by 1.  
- positive values make the hit to be put into an existing cluster: the unit interval is divided in `l` sub-intervals, the one that the action-value is binned to becomes the particle slice index.

## DDPG Agent

The Deep Deterministic Policy Gradient agent is comprised of two networks that train simultaneously. An _Actor_ network takes as input a CaloHit state observation and outputs and action-value associated to a specific action. A _Critic_  network receives the CaloHit state concatenated with the action-value and returns the Q-value related to the action.

## Reward

Training a RL model requires a total reward that the Agent tries to maximize throughout an entire episode. The reward associated to a single action for this model is the Earth Movers Distance (EMD) between the current predicted slice where the CaloHit is appended and the real slice where the CaloHit belongs to.  

The [EnergyFlow](https://github.com/pkomiske/EnergyFlow) Python package is used to compute the EMD distance between slices. EMD distance asks a first energy parameter and then a set of `gdim` coordinates for each point. By default the `gdim-`euclidean distance is used to compute the distance in the ground space. Refer to [this](https://github.com/pkomiske/EnergyFlow/blob/57ae31066de20b024fa3a48bf8318cdd88fad6c9/energyflow/emd.py#L115) for the function documentation.

SliceRL feeds a tuple of `(E,x,z)` coordinates to the EDM function. First parameter is the total energy, second is the sum of `x` coordinates of all the CaloHits in the slice, the last is the sum of coordinates in the specific U/V/W plane considered.

Note: For EMD to be a distance metric, the `R` parameter has to be greater than half of the maximum distance between CaloHits. For protoDUNE we have that parameter must be greater than about 600 millimeters for the (x,U/V/W) planes (0.6 in the code).

## Notes

Input lenghts are expressed in millimeters, while energies in ADCs. In order to have smaller number and gradients involved in the slicing computations, rescaling factors of 1e3 and 1e2 are employed for lenghts and energies respectively.
