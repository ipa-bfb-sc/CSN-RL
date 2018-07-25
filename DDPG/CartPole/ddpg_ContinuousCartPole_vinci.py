import numpy as np
import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from DDPG.ddpg import DDPGAgent

from common.memory import SimpleMemory, SequentialMemory
from common.random import OrnsteinUhlenbeckProcess,GaussianWhiteNoiseProcess
from common.callbacks import TestLogger, TrainEpisodeLogger, TrainIntervalLogger, Visualizer, CallbackList, FileLogger
from common.utils.networks import simple_actor, simple_critic
from common.utils.env import populate_env
from common.runtime.experiment import DefaultExperiment

import environments
from datetime import datetime
timenow = datetime.now().strftime('%Y-%m-%d %H:%M:%S')



gym.undo_logger_setup()

ENV_NAME = 'ContinuousCartPole-v0'
#ENV_NAME = 'Pendulum-v0'
#ENV_NAME = 'PlaneBall-v0'
#ENV_NAME = 'GazeboCircuit2TurtlebotLidarNn-v1'

env_old = gym.make(ENV_NAME)

# Experiment
experiment = DefaultExperiment()

with experiment:
    # Get the environment
    # And populate it with useful metadata
    env = populate_env(env_old)
    np.random.seed(123)
    env.seed(123)

    # Build the actor and the critic
    actor = simple_actor(env)
    critic = simple_critic(env)

    # Memory
    memory = SimpleMemory(env=env, limit=100000)
    #memory = SequentialMemory(limit=10000, window_length=1)
    # Noise
    #random_process = OrnsteinUhlenbeckProcess(size=env.action_space.dim, theta=.15, mu=0., sigma=3.)
    random_process = GaussianWhiteNoiseProcess()


    callback1 = FileLogger(filepath='save/history1_{}'.format(timenow), interval=1)

    # Agent
    agent = DDPGAgent(
        experiment=experiment,
        actor=actor,
        critic=critic,
        env=env,
        memory=memory,
        #random_process=random_process
    )
    agent.compile()

    history = agent.train(
        env=env,
        episodes=3000,
        render=False,
        callbacks=[callback1],
        verbosity=2,
        plots=False)

    test = agent.test(env=env)
    agent.save_weights('save/ddpg_ContinuousCartpole_weights.h5f', overwrite=True)
    #print(history.history.keys())


'''
    agent._run(
             episodes=20,
             train=True,
             render=False,
             exploration=True,
             plots=False,
             tensorboard=False,
             callbacks=callback1,
             verbosity=2,
             action_repetition=1,
             nb_max_episode_steps=None,
             log_interval=10000)
'''
