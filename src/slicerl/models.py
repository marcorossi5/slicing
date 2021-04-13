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
    Input,
    Dense,
    Flatten,
    Activation,
    Reshape,
    Concatenate,
    BatchNormalization,
    ReLU,
    Dropout,
    Masking,
    Permute
)

from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard

from hyperopt import STATUS_OK
from time import time

import pprint, json

#----------------------------------------------------------------------
def Block(nb_units_layers, name):
    model = Sequential(name=name)
    for i, nb_units in enumerate(nb_units_layers):
        model.add(Dense(nb_units, name=f"{name}_dense_{i}"))
        # model.add(BatchNormalization(name=f"{name}_bnorm_{i}"))
        model.add(Activation('relu', name=f"{name}_relu_{i}"))
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
    """Construct the actor model used by the DDPG. Outputs the action. """
    # TODO: include the possibility to process memory experience
    nmemory, ncalohits, nfeatures = input_dim
    nlatent = 256

    # declare layers
    obs_state       = Input(shape=input_dim, name='a_obs_input')

    mask            = Masking(mask_value=0., name='a_mask')

    permute         = Permute((1,3,2), name='a_permute')
    nb_units_dnsz   = [nlatent] # [1024, 512, nlatent]
    a_downsize      = Block(nb_units_dnsz, "a_downsize")

    nb_units_local  = [64,64]
    a_local         = Block(nb_units_local, "a_local")

    nb_units_global = [64,128,256]
    a_global        = Block(nb_units_global, "a_global")

    concat          = Concatenate(-1, name='a_concat_local_global')
    nb_units_cat    = [128,64,32,16]
    a_cat           = Block(nb_units_cat, "a_cat")

    nb_units_upsz   = [ncalohits] # [512, 1024, ncalohits]
    a_upsize        = Block(nb_units_upsz, "a_upsize") 

    a_final         = Dense(1, name='a_final_dense')
    activation      = Activation('sigmoid', name='a_activation')

    shape           = (ncalohits,)
    reshape         = Reshape(shape, name='a_reshape')

    # declare forward pass
    masked_state = mask(obs_state)

    dnsz_state  = permute(a_downsize(permute(masked_state)))

    local_state = a_local(dnsz_state)

    global_state = a_global(local_state)
    
    global_state = tf.math.reduce_max(global_state, axis=-2, keepdims=True, name='a_max_pool')
    global_state = tf.repeat(global_state, nlatent, axis=-2, name='a_repeat')
    
    cat_state = a_cat(concat([local_state, global_state]))

    # if memory window length > 1 at this point it would break
    # we would have to involve also memory axis in computation
    up_state    = permute(a_upsize(permute(cat_state)))

    final_state = reshape(activation(a_final(up_state)))

    model = Model(inputs=obs_state, outputs=final_state, name="Actor")
    model.summary()
    return model

#----------------------------------------------------------------------
def build_critic_model(hps, input_dim):
    """
    Construct the critic model used by the DDPG. Judges the goodness of the
    predicted action
    """
    # TODO: think about adding a small epsilon to the action to mask it properly
    # In any case there's the random_process that sums to the action, so padded
    # calohits are not zero at the end, that's why the mask here could be useless
    nmemory, ncalohits, nfeatures = input_dim
    nlatent = 256

    action_input    = Input(shape=(ncalohits,), name='c_act_input')
    obs_input       = Input(shape=input_dim, name='c_obs_input')

    reshape_act     = Reshape((ncalohits,1), name='c_reshape_act')
    reshape_obs     = Reshape((ncalohits,4), name='c_reshape_obs') # works only if nmemory == 1

    concat          = Concatenate(-1, name='c_concat_obs_act')

    permute         = Permute((2,1), name='c_permute')
    nb_units_dnsz   = [nlatent] # [1024, 512, nlatent]
    c_downsize      = Block(nb_units_dnsz, "c_downsize")

    nb_units_local  = [32, 128, 512]
    c_local         = Block(nb_units_local, "c_local")

    flatten         = Flatten(name='c_flatten')

    nb_units_global = [64, 16, 8]
    c_global        = Block(nb_units_global, "c_global")

    dropout         = Dropout(hps['dropout'], name='c_dropout')
    c_final         = Dense(1, name='c_final_dense')
    activation      = Activation('sigmoid', name='c_activation')

    x = concat([reshape_obs(obs_input), reshape_act(action_input)])

    dnsz_state  = permute(c_downsize(permute(x)))
    local_state = c_local(dnsz_state)
    
    local_state = flatten( tf.math.reduce_max(local_state, axis=-2, name='c_max_pool') )

    global_state = c_global(local_state)

    policy = activation(c_final(dropout(global_state)))

    model = Model(inputs=[action_input, obs_input], outputs=policy, name="Critic")
    model.summary()
    return model, action_input

#----------------------------------------------------------------------
def build_ddpg(hps, input_dim):
    """Create a DDPG agent to be used on pandora inputs."""
    print('[+] Constructing DDPG agent, model setup:')
    pprint.pprint(hps)
    nmemory, ncalohits, nfeatures = input_dim

    # set up the agent
    K.clear_session()
    actor_model = build_actor_model(hps['actor'], input_dim)
    critic_model, action_input = build_critic_model(hps['critic'], input_dim)

    memory = SequentialMemory(limit=500000, window_length=1)
    # random_process = OrnsteinUhlenbeckProcess(size=ncalohits, theta=.15, mu=0., sigma=.1)
    random_process = GaussianWhiteNoiseProcess(size=ncalohits, mu=0., sigma=0.3)
    agent = DDPGAgentSlicerl(actor=actor_model, critic=critic_model,
                          critic_action_input=action_input, nb_actions=ncalohits,
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
        ], dtype=np.float32
    ).reshape(1,4)
    low = np.repeat(low, env_setup['max_hits'], 0)

    high = np.array(
        [
            500.,                   # E
            0.37260447692861504,    # x
            0.91702,                # z
            150,                    # cluster idx
        ], dtype=np.float32
    ).reshape(1,4)
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
        agent = build_ddpg(agent_setup, (agent_setup['memory'],) + slicerl_env.observation_space.shape)

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
