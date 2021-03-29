# This file is part of SliceRL by M. Rossi

from slicerl.SlicerlEnv import SlicerlEnvContinuous
import numpy as np

from slicerl.tools import get_window_width, mass
from slicerl.AgentSlicerl import DDPGAgentSlicerl
from slicerl.read_data import Events
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, LSTM, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard

from hyperopt import STATUS_OK
from time import time

import pprint, json

#----------------------------------------------------------------------
def build_actor_model(hps, input_dim):
    """Construct the actor model used by the DDPG. Outputs the action"""
    model = Sequential()
    if hps['architecture']=='Dense':
        model.add(Flatten(input_shape=(1,) + input_dim))
        for i in range(hps['nb_layers']):
            model.add(Dense(hps['nb_units']))
            model.add(Activation('relu'))
        if hps['dropout']>0.0:
            model.add(Dropout(hps['dropout']))
        model.add(Dense(1))
        model.add(Activation('linear'))
    elif hps['architecture']=='LSTM':
        model.add(LSTM(hps['nb_units'], input_shape = (1,max(input_dim)),
                       return_sequences=not (hps['nb_layers']==1)))
        for i in range(hps['nb_layers']-1):
            model.add(LSTM(hps['nb_units'],
                           return_sequences=not (i+2==hps['nb_layers'])))
        if hps['dropout']>0.0:
            model.add(Dropout(hps['dropout']))
        model.add(Dense(1))
        model.add(Activation('tanh'))
    model.summary()
    return model

#----------------------------------------------------------------------
def build_critic_model(hps, input_dim):
    """
    Construct the critic model used by the DDPG. Judge the goodness of the
    predicted action
    """
    action_input = Input(shape=(1,), name='action_input')
    obs_input = Input(shape=(1,)+input_dim, name='observation_input')
    flattened_obs = Flatten()(obs_input)
    x = Concatenate()([action_input, flattened_obs])
    if hps['architecture']=='Dense':
        for i in range(hps['nb_layers']):
            x = Dense(hps['nb_units'])(x)
            x = Activation('relu')(x)
        if hps['dropout']>0.0:
            x = Dropout(hps['dropout'])(x)
        x = Dense(1)(x)
        x = Activation('linear')(x)
    elif hps['architecture']=='LSTM':
        raise NotImplementedError("LSTM for critic not implemented")
        x = LSTM(hps['nb_units'], input_shape = (1,max(input_dim)),
                       return_sequences=not (hps['nb_layers']==1))(x)
        for i in range(hps['nb_layers']-1):
            x = LSTM(hps['nb_units'],
                           return_sequences=not (i+2==hps['nb_layers']))(x)
        if hps['dropout']>0.0:
            x = Dropout(hps['dropout'])(x)
        x = Dense(1)(x)
        x = Activation('linear')(x)
    model = Model(inputs=[action_input, obs_input], outputs=x)
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
    random_process = OrnsteinUhlenbeckProcess(size=1, theta=.15, mu=0., sigma=.3)
    agent = DDPGAgentSlicerl(actor=actor_model, critic=critic_model,
                          critic_action_input=action_input, nb_actions=1,
                          memory=memory, nb_steps_warmup_actor=100,
                          nb_steps_warmup_critic=100, target_model_update=1e-2,
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
    # if there is a state_dim variable, set up LundCoordinates accordingly
    # unless we are doing a scan (in which case it needs to be done later)
    env_setup = res.get("slicerl_env")
    return res

#----------------------------------------------------------------------
def loss_calc(dqn, fn, nev):
    pass
    # reader = Events(fn, nev)
    # subtractor = dqn.subtractor()
    # self.jet_pairs_list = reader.values()
    # for jet_pairs in reader:
    #     if len(jet_pairs)==0:
    #         continue
    #     events = Event(jet_pairs)
    #     for event in events:
    #         if
    #     reader_sig = Jets(fn_sig, nev) # load validation set
    #     reader_bkg = Jets(fn_bkg, nev) # load validation set
    #     groomed_jets_sig = []
    #     for jet in reader_sig.values():
    #         groomed_jets_sig.append(dqn.groomer()(jet))
    #     masses_sig = np.array(mass(groomed_jets_sig))
    #     lower, upper, median = get_window_width(masses_sig)
    #     groomed_jets_bkg = []
    #     for jet in reader_bkg.values():
    #         groomed_jets_bkg.append(dqn.groomer()(jet))
    #     masses_bkg = np.array(mass(groomed_jets_bkg))
    #     # calculate the loss function
    #     count_bkg = ((masses_bkg > lower) & (masses_bkg < upper)).sum()
    #     frac_bkg = count_bkg/float(len(masses_bkg))
    #     loss = abs(upper-lower)/5 + abs(median-massref) + frac_bkg*20
    #     return loss, (lower,upper,median)

#----------------------------------------------------------------------
def load_environment(env_setup):
    slicerl_env = SlicerlEnvContinuous(env_setup,
                low=np.array([0., -0.37260447692861504, -0.35284, 0.], dtype=np.float32),
                high=np.array([500., 0.37260447692861504, 0.91702, 150.], dtype=np.float32)
                                     )
    # slicerl_env = SlicerlEnvContinuous(env_setup,
    #                     flow=np.array([0., -0.37260447692861504, -0.35284], dtype=np.float32),
    #                     fhigh=np.array([500., 0.37260447692861504, 0.91702], dtype=np.float32),
    #                     ihigh=np.array([150])
    #                     )
    return slicerl_env

#----------------------------------------------------------------------
def build_and_train_model(slicerl_agent_setup, slicerl_env=None):
    """Run a test model"""

    if slicerl_env is None:
        env_setup = slicerl_agent_setup.get('slicerl_env')
        slicerl_env = load_environment(env_setup)

    agent_setup = slicerl_agent_setup.get('rl_agent')
    ddpg = build_ddpg(agent_setup, slicerl_env.observation_space.shape)

    logdir = '%s/logs/{}'.format(time()) % slicerl_agent_setup['output']
    print(f'[+] Constructing tensorboard log in {logdir}')
    tensorboard = TensorBoard(log_dir=logdir)

    print('[+] Fitting DDPG agent...')
    r = ddpg.fit(slicerl_env, nb_steps=agent_setup['nstep'],
                visualize=False, verbose=1, callbacks=[tensorboard])

    # compute nominal reward after training
    median_reward = np.median(r.history['episode_reward'])
    print(f'[+] Median reward: {median_reward}')

    # After training is done, we save the final weights.
    if not slicerl_agent_setup['scan']:
        weight_file = '%s/weights.h5' % slicerl_agent_setup['output']
        print(f'[+] Saving weights to {weight_file}')
        ddpg.save_weights(weight_file, overwrite=True)

        # save the model architecture in json
        model_file = '%s/model' % slicerl_agent_setup['output']
        actor_file = '%s_actor.json' % model_file
        critic_file = '%s_critic.json' % model_file        
        print(f'[+] Saving model to {actor_file}, {critic_file}')
        with open(actor_file, 'w') as outfile:
            json.dump(ddpg.actor.to_json(), outfile)
        with open(critic_file, 'w') as outfile:
            json.dump(ddpg.critic.to_json(), outfile)

    if slicerl_agent_setup['scan']:
        # compute a metric for training set (TODO: change to validation)
        raise ValueError('SCAN LOSS FCT NOT IMPLEMENTED YET')
        loss, window = loss_calc(ddpg, env_setup['val'],  env_setup['nev_val'])
        print(f'Loss function for scan = {loss}')
        res = {'loss': loss, 'reward': median_reward, 'status': STATUS_OK}
    else:
        res = ddpg
    return res
