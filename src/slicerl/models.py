# This file is part of SliceRL by M. Rossi

from slicerl.SlicerlEnv import SlicerlEnvDiscrete, SlicerlEnvContinuous
import numpy as np

from slicerl.tools import get_window_width, mass
from slicerl.AgentSlicerl import DQNAgentSlicerl, DDPGAgentSlicerl
from slicerl.read_data import Events
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess, GaussianWhiteNoiseProcess

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense,
    Activation,
    Flatten,
    Dropout,
    Input,
    Concatenate,
    Reshape,
    BatchNormalization,
    ReLU
)
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard

from hyperopt import STATUS_OK
from time import time

import pprint, json

#----------------------------------------------------------------------
def Block(hps, name, activation, output_dim=1):
    model = Sequential(name=name)
    for i in range(hps['nb_layers']):
        model.add(Dense(hps['nb_units']))
        model.add(Activation('relu'))
    if hps['dropout']>0.0:
        model.add(Dropout(hps['dropout']))
    model.add(Dense(output_dim))
    model.add(Activation(activation))
    return model

#----------------------------------------------------------------------
def build_model(hps, input_dim, k, output_dim):
    """Construct the actor model used by the DDPG. Outputs the action"""
    input_state     = Input(shape=(1,) + input_dim, name='actor_input')    
    graphBlock      = Block(hps, 'GraphBlock', 'linear')
    aggregatorBlock = Block(hps, 'AggregatorBlock', 'linear')
    finalBlock      = Block(hps, 'FinalBlock', 'softmax', output_dim=output_dim)
    
    reshaped_state = Reshape(input_dim)(input_state)
    c, notc = tf.split(reshaped_state, [1,k], -2)
    c = Reshape((6,))(c)
    
    neigh_state = graphBlock(notc)
    neigh_state = Reshape((k,))(neigh_state)
    aggr_state = aggregatorBlock(neigh_state)

    cat_state = Concatenate(axis=-1)([c, aggr_state])
    out = finalBlock(cat_state)

    model = Model(inputs=input_state, outputs=out, name="Actor")
    model.summary()
    return model

#----------------------------------------------------------------------
def build_dqn(hps, input_dim, k, output_dim):
    """Create a DQN agent to be used on lund inputs."""

    print('[+] Constructing DQN agent, model setup:')
    pprint.pprint(hps)

    # set up the DQN agent
    K.clear_session()
    model = build_model(hps['dqn'], input_dim, k, output_dim)
    memory = SequentialMemory(limit=500000, window_length=1)
    if hps["policy"]=="boltzmann":
        policy = BoltzmannQPolicy()
    elif hps["policy"]=="epsgreedyq":
        policy = EpsGreedyQPolicy()
    else:
        raise ValueError("Invalid policy: %s"%hps["policy"])
    duelnet = hps["enable_dueling_network"]
    doubdqn = hps["enable_double_dqn"]
    agent = DQNAgentSlicerl(model=model, nb_actions=output_dim,
                          enable_dueling_network=duelnet,
                          enable_double_dqn=doubdqn,
                          memory=memory, nb_steps_warmup=500,
                          target_model_update=1e-2, policy=policy)

    if hps['optimizer'] == 'Adam':
        opt = Adam(lr=hps['learning_rate'])
    elif hps['optimizer']  == 'SGD':
        opt = SGD(lr=hps['learning_rate'])
    elif hps['optimizer'] == 'RMSprop':
        opt = RMSprop(lr=hps['learning_rate'])
    elif hps['optimizer'] == 'Adagrad':
        opt = Adagrad(lr=hps['learning_rate'])

    agent.compile(opt, metrics=['mae'])

    return agent

#----------------------------------------------------------------------
def build_actor_model(hps, input_dim):
    """Construct the actor model used by the DDPG. Outputs the action"""
    input_state = Input(shape=(1,) + input_dim, name='actor_input')

    nb_units_local  = [64, 64]
    dense_local     = [Dense(nb_units, name=f'a_dense_local_{i}') for i, nb_units in enumerate(nb_units_local)]
    bn_local        = [BatchNormalization(name=f'a_bnorm_local_{i}') for i in range(len(nb_units_local))]
    relu_local      = [ReLU(name=f'a_relu_local_{i}') for i in range(len(nb_units_local))]

    nb_units_global = [64, 128, 1024]
    dense_global    = [Dense(nb_units, name=f'a_dense_global_{i}') for i, nb_units in enumerate(nb_units_global)]
    bn_global       = [BatchNormalization(name=f'a_bnorm_global_{i}') for i in range(len(nb_units_global))]
    relu_global     = [ReLU(name=f'a_relu_global_{i}') for i in range(len(nb_units_global))]

    concat          = Concatenate(-1, name='a_concat_local_global')

    nb_units_cat    = [512, 256, 128, 64]
    dense_cat       = [Dense(nb_units, name=f'a_dense_cat_{i}') for i, nb_units in enumerate(nb_units_cat)]
    bn_cat          = [BatchNormalization(name=f'a_bnorm_cat_{i}') for i in range(len(nb_units_cat))]
    relu_cat        = [ReLU(name=f'relu_cat_{i}') for i in range(len(nb_units_cat))]

    dense_final     = Dense(1, name='a_dense_final')
    activation      = Activation('sigmoid', name='a_activation')
    reshape         = Reshape((input_dim[0],), name='a_reshape')

    x = input_state
    for dense, bn, relu in zip(dense_local, bn_local, relu_local):
        x = relu(bn(dense(x)))
    
    y = x
    for dense, bn, relu in zip(dense_global, bn_global, relu_global):
        y = relu(bn(dense(y)))
    
    y = tf.math.reduce_max(y, axis=-2, keepdims=True, name='a_max_pool')
    y = tf.repeat(y, input_dim[0], axis=-2, name='a_repeat')
    
    cat_state = concat([x,y])
    for dense, bn, relu in zip(dense_cat, bn_cat, relu_cat):
        cat_state = relu(bn(dense(cat_state)))

    final_state = reshape(activation(dense_final(cat_state)))
    
    model = Model(inputs=input_state, outputs=final_state, name="Actor")
    model.summary()
    return model

#----------------------------------------------------------------------
def build_critic_model(hps, input_dim):
    """
    Construct the critic model used by the DDPG. Judge the goodness of the
    predicted action
    """
    obs_input       = Input(shape=(1,)+input_dim, name='observation_input')
    action_input    = Input(shape=(input_dim[0],), name='action_input')

    reshape_obs     = Reshape((input_dim[0],5), name='c_reshape_obs')
    reshape_act     = Reshape((input_dim[0],1), name='c_reshape_act')

    concat          = Concatenate(-1, name='c_concat_obs_act')

    nb_units_local  = [64, 256, 1024]
    dense_local     = [Dense(nb_units, name=f'c_dense_local_{i}') for i, nb_units in enumerate(nb_units_local)]
    bn_local        = [BatchNormalization(name=f'c_bnorm_local_{i}') for i in range(len(nb_units_local))]
    relu_local      = [ReLU(name=f'c_relu_local_{i}') for i in range(len(nb_units_local))]

    flatten         = Flatten(name='c_flatten')

    nb_units_global = [256, 64, 32]
    dense_global    = [Dense(nb_units, name=f'c_dense_global_{i}') for i, nb_units in enumerate(nb_units_global)]
    bn_global       = [BatchNormalization(name=f'c_bnorm_global_{i}') for i in range(len(nb_units_global))]
    relu_global     = [ReLU(name=f'c_relu_global_{i}') for i in range(len(nb_units_global))]

    dropout         = Dropout(hps['dropout'], name='c_dropout')
    dense_final     = Dense(1, name='c_dense_final')
    activation      = Activation('sigmoid', name='c_activation')

    x = concat([reshape_obs(obs_input), reshape_act(action_input)])
    for dense, bn, relu in zip(dense_local, bn_local, relu_local):
        x = relu(bn(dense(x)))
    
    x = flatten( tf.math.reduce_max(x, axis=-2, name='c_max_pool') )

    for dense, bn, relu in zip(dense_global, bn_global, relu_global):
        x = relu(bn(dense(x)))

    policy = activation(dense_final(dropout(x)))

    model = Model(inputs=[action_input, obs_input], outputs=policy, name="Critic")
    model.summary()
    return model, action_input

#----------------------------------------------------------------------
def build_ddpg(hps, input_dim):
    """Create a DDPG agent to be used on pandora inputs."""

    print('[+] Constructing DDPG agent, model setup:')
    pprint.pprint(hps)

    # set up the agent
    K.clear_session()
    actor_model = build_actor_model(hps['actor'], input_dim)
    critic_input_dim = input_dim
    critic_model, action_input = build_critic_model(hps['critic'], critic_input_dim)

    memory = SequentialMemory(limit=500000, window_length=1)
    # random_process = OrnsteinUhlenbeckProcess(size=input_dim[0], theta=.15, mu=0., sigma=.1)
    random_process = GaussianWhiteNoiseProcess(size=input_dim[0], mu=0., sigma=0.3)
    agent = DDPGAgentSlicerl(actor=actor_model, critic=critic_model,
                          critic_action_input=action_input, nb_actions=input_dim[0],
                          memory=memory, nb_steps_warmup_actor=34,
                          nb_steps_warmup_critic=34, target_model_update=1e-2,
                          random_process=random_process)

    if hps['optimizer'] == 'Adam':
        opt = Adam(lr=hps['learning_rate'])
    elif hps['optimizer']  == 'SGD':
        opt = SGD(lr=hps['learning_rate'])
    elif hps['optimizer'] == 'RMSprop':
        opt = RMSprop(lr=hps['learning_rate'])
    elif hps['optimizer'] == 'Adagrad':
        opt = Adagrad(lr=hps['learning_rate'])

    agent.compile(opt, metrics=['mae'])

    return agent

#----------------------------------------------------------------------
def load_runcard(runcard):
    """Read in a runcard json file and set up dimensions correctly."""
    with open(runcard,'r') as f:
        res = json.load(f)
    env_setup = res.get("slicerl_env")
    return res

#----------------------------------------------------------------------
def loss_calc(dqn, fn, nev):
    pass

#----------------------------------------------------------------------
def load_environment(env_setup):
    low = np.array(
        [
            0.,                     # E
            -0.37260447692861504,   # x
            -0.35284,               # z
            0.,                     # cluster idx
            0.,                     # status
        ], dtype=np.float32
    ).reshape(1,5)
    low = np.repeat(low, env_setup['max_hits'], 0)

    high = np.array(
        [
            500.,                   # E
            0.37260447692861504,    # x
            0.91702,                # z
            150,                    # cluster idx
            128.,                   # status
        ], dtype=np.float32
    ).reshape(1,5)
    high = np.repeat(high, env_setup['max_hits'], 0)

    if env_setup["discrete"]:
        slicerl_env = SlicerlEnvDiscrete(env_setup, low=low, high=high)
    else:
        slicerl_env = SlicerlEnvContinuous(env_setup, low=low, high=high)
    return slicerl_env

#----------------------------------------------------------------------
def build_and_train_model(slicerl_agent_setup, slicerl_env=None):
    """Run a test model"""

    if slicerl_env is None:
        env_setup = slicerl_agent_setup.get('slicerl_env')
        slicerl_env = load_environment(env_setup)

    agent_setup = slicerl_agent_setup.get('rl_agent')
    if slicerl_env.discrete:
        agent = build_dqn(agent_setup, slicerl_env.observation_space.shape, slicerl_env.max_hits, slicerl_env.nbins)
    else:
        agent = build_ddpg(agent_setup, slicerl_env.observation_space.shape)

    logdir = '%s/logs/{}'.format(time()) % slicerl_agent_setup['output']
    print(f'[+] Constructing tensorboard log in {logdir}')
    tensorboard = TensorBoard(log_dir=logdir)

    print('[+] Fitting agent...')
    r = agent.fit(slicerl_env, nb_steps=agent_setup['nstep'],
                visualize=False, verbose=1, callbacks=[tensorboard],
                log_interval=128, nb_max_episode_steps=128)

    # compute nominal reward after training
    median_reward = np.median(r.history['episode_reward'])
    print(f'[+] Median reward: {median_reward}')

    # After training is done, we save the final weights.
    if not slicerl_agent_setup['scan']:
        weight_file = '%s/weights.h5' % slicerl_agent_setup['output']
        print(f'[+] Saving weights to {weight_file}')
        agent.save_weights(weight_file, overwrite=True)

        # save the model architecture in json
        model_file = '%s/model' % slicerl_agent_setup['output']
        if slicerl_env.discrete:
            with open(model_file, 'w') as outfile:
                json.dump(agent.model.to_json(), outfile)
        else:
            actor_file = '%s_actor.json' % model_file
            critic_file = '%s_critic.json' % model_file        
            print(f'[+] Saving model to {actor_file}, {critic_file}')
            with open(actor_file, 'w') as outfile:
                json.dump(agent.actor.to_json(), outfile)
            with open(critic_file, 'w') as outfile:
                json.dump(agent.critic.to_json(), outfile)

    if slicerl_agent_setup['scan']:
        # compute a metric for training set (TODO: change to validation)
        raise ValueError('SCAN LOSS FCT NOT IMPLEMENTED YET')
        loss, window = loss_calc(agent, env_setup['val'],  env_setup['nev_val'])
        print(f'Loss function for scan = {loss}')
        res = {'loss': loss, 'reward': median_reward, 'status': STATUS_OK}
    else:
        res = agent
    return res
