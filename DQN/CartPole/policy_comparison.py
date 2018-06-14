import numpy as np
import gym
import json

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam, SGD

#from DQN.dqn import DQNAgent
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from matplotlib import pyplot
from keras.models import model_from_json

ENV_NAME = 'CartPole-v0'
env = gym.make(ENV_NAME)
np.random.seed(555)
env.seed(555)
nb_actions = env.action_space.n

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())



with open('save/history1_2018-06-13 12:34:33', 'r') as f:
    pp1_1 = json.load(f)
    f.close()

with open('save/history1_2018-06-13 14:11:42', 'r') as f:
    pp1_2 = json.load(f)
    f.close()

with open('save/history1_2018-06-13 14:30:07', 'r') as f:
    pp1_3 = json.load(f)
    f.close()

duration1 = sum(pp1_1['duration'])
duration2 = sum(pp1_2['duration'])
duration3 = sum(pp1_3['duration'])
print('Duration1:{}, Duration2:{}, Duration3:{}'.format(duration1,duration2,duration3))

pyplot.subplot(2, 1, 1)
pyplot.xlabel('episodes')
pyplot.ylabel('rewards per episode')
pyplot.plot(pp1_1['episode_reward'], 'r', label='EpsDisGreedy, 1-0.01')
pyplot.plot(pp1_2['episode_reward'], 'g', label='EpsDisGreedy with random action in 1000 warm_up steps, 1-0.01')
pyplot.plot(pp1_3['episode_reward'], 'b', label='EpsDisGreedywith random action in 1000 warm_up steps, 1-0.001')
pyplot.legend()

pyplot.subplot(2, 1, 2)
pyplot.xlabel('total steps')
pyplot.ylabel('rewards per episode')
pyplot.plot(pp1_1['nb_steps'], pp1_1['episode_reward'], 'r', label='EpsDisGreedy, 1-0.01')
pyplot.plot(pp1_2['nb_steps'], pp1_2['episode_reward'], 'g', label='EpsDisGreedy with random action in 1000 warm_up steps, 1-0.01')
pyplot.plot(pp1_3['nb_steps'], pp1_3['episode_reward'], 'b', label='EpsDisGreedywith random action in 1000 warm_up steps, 1-0.001')

pyplot.legend()

pyplot.show()
