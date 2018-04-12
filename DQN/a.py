from __future__ import print_function
import numpy as np
import gym
from gym.spaces import Discrete, Box


# DQN for Reinforcement Learning
# by Qin Yongliang
# 2017 01 11

def continuous_actions(env):
    if isinstance(env.action_space, Box):
        pass

    split = 9

    dims = env.action_space.shape[0]
    action_count = split ** dims

    low = env.action_space.low
    high = env.action_space.high

    itvl = high - low

    def d2c(index):
        idx = index
        continuous = []
        for i in range(dims):
            rem = idx % split
            idx = int(idx / split)
            continuous.append(rem)

        cont = (np.array(continuous, dtype='float32') / (split - 1)) * itvl + low
        #print('NEW:')
        #print(cont)
        return cont

    return action_count, d2c


# run episode with some policy of some agent, and collect the rewards
# well the usual gym stuff
def do_episode_collect_trajectory(agent, env, max_steps, render=True, feed=True, realtime=False, use_best=False):
    observation = env.reset()  # s1
    last_observation = observation

    agent.wakeup()  # notify the agent episode starts
    total_reward = 0
    for t in range(max_steps):
        combined_observation = np.hstack([last_observation, observation])
        last_observation = observation

        action = agent.act(combined_observation, use_best=use_best)  # a1

        if isinstance(env.action_space, Box):
            action_count, d2c = continuous_actions(env)
            actual_action = d2c(action)
        else:
            actual_action = action

        # s2, r1,
        observation, reward, done, _info = env.step(actual_action)

        # d1
        isdone = 1 if done else 0
        total_reward += reward

        if feed:
            agent.feed_immediate_data((combined_observation, action, reward, isdone))

        if render and (t % 10 == 0 or realtime == True): env.render()

        if done:
            break
    print('episode done in', t, 'steps, total reward', total_reward)

    return


# keras boilerplate: the simplest way to neural networking
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import keras
from math import *
import keras.backend as K
import time


# our neural network agent.
class nnagent(object):
    def __init__(self, num_of_actions, num_of_observations, discount_factor, optimizer, epsilon=-1, ):
        # agent database
        self.observations = np.zeros((0, num_of_observations))
        self.actions = np.zeros((0, num_of_actions))
        self.rewards = np.zeros((0, 1))
        self.isdone = np.zeros((0, 1))

        # agent property
        self.num_of_actions = num_of_actions
        self.num_of_observations = num_of_observations
        self.discount_factor = discount_factor

        self.epsilon = epsilon  # epsilon-greedy per David Silver's Lecture and DeepMind paper.

        # -----------------------------
        # Deep-Q-Network

        #from keras.regularizers import l2, activity_l2

        input_shape = num_of_observations
        inp = Input(shape=(input_shape,))
        i = inp

        i = Dense(16, activation='tanh')(i)
        i = Dense(32, activation='tanh')(i)
        i = Dense(64, activation='tanh')(i)

        i = Dense(num_of_actions)(i)
        out = i
        qfunc = Model(input=inp, output=out)
        self.qfunc = qfunc

        # ------------------------------

        # ------------------------------
        # DQN trainer
        s1 = Input(shape=(input_shape,))
        a1 = Input(shape=(num_of_actions,))
        r1 = Input(shape=(1,))
        isdone = Input(shape=(1,))

        qs2 = Input(shape=(num_of_actions,))  # qs2 is precalc-ed

        q_prediction = qfunc(s1)
        # the q values we predicted for the given state.

        q_s1_a1 = merge([q_prediction, a1],
                        mode=(lambda x: K.sum(x[0] * x[1], axis=-1, keepdims=True)),
                        output_shape=(1,))

        def calc_target(x):
            qs2 = x[0]  # q value of next state
            r1 = x[1]
            isdone = x[2]
            return (K.max(qs2, axis=-1, keepdims=True) * discount_factor * (1 - isdone) + r1)

        q_target = merge([qs2, r1, isdone],
                         mode=calc_target, output_shape=(1,))

        # target = sum of [immediate reward after action a] and [q values predicted for next state, discounted]. target is a better approximation of q function for current state, so we use it as the training target.

        def mse(x):
            return K.mean((x[0] - x[1]) ** 2, axis=-1, keepdims=True)

        q_loss = merge([q_target, q_s1_a1],
                       mode=mse, output_shape=(1,), name='q_loss')
        # what we meant: q_loss = (q_target - q_prediction)**2

        qtrain = Model(input=[s1, a1, r1, isdone, qs2], output=q_loss)

        def pass_thru(y_true, y_pred):
            return K.mean(y_pred, axis=-1)

        qtrain.compile(loss=pass_thru, optimizer=optimizer)

        # -----------------------------
        self.qfunc = qfunc
        self.qtrain = qtrain

        print('agent Initialized with', num_of_observations, 'dim input and', num_of_actions, 'dim output.')
        print('discount_factor', discount_factor)
        print('model architechture:')
        qfunc.summary()
        print('trainer architechture:')
        qtrain.summary()

    # act one step base on observation
    def act(self, observation, use_best=False):
        qfunc = self.qfunc
        epsilon = self.epsilon  # greedy factor

        observation = observation.reshape((1, len(observation)))

        # observation is a vector
        qvalues = qfunc.predict([observation])[0]

        # for qfunc:
        # with probability epsilon we act randomly:
        if (self.epsilon > np.random.rand(1)) and use_best == False:
            action_index = np.random.choice(len(qvalues))
        else:
            # with probability 1-epsilon we act greedy:
            action_index = qvalues.argmax()

        # print(action_index)

    def wakeup(self):
        # clear states
        pass

    # after playing for one(or whatever) episode, we could feed the agent with data.
    def feed_episodic_data(self, episodic_data):
        observations, actions, rewards, isdone = episodic_data

        actions = np.array(actions)
        rewards = np.array(rewards).reshape((-1, 1))
        isdone = np.array(isdone).reshape((-1, 1))

        # IMPORTANT: convert actions to their one-hot representations
        def one_hot(tensor, classes):
            heat = np.zeros(tensor.shape + (classes,))
            for i in range(classes):
                heat[..., i] = tensor[...] == i
            return heat

        onehot_actions = one_hot(actions, self.num_of_actions)

        # add to agent's database
        self.observations = np.vstack((self.observations, observations))
        self.actions = np.vstack((self.actions, onehot_actions))
        self.rewards = np.vstack((self.rewards, rewards))
        self.isdone = np.vstack((self.isdone, isdone))

    def feed_immediate_data(self, immediate_data):
        observation, action, rewards, isdone = immediate_data

        action = np.array(action)
        reward = np.array(rewards).reshape((-1, 1))
        isdone = np.array(isdone).reshape((-1, 1))

        # IMPORTANT: convert actions to their one-hot representations
        def one_hot(tensor, classes):
            heat = np.zeros(tensor.shape + (classes,))
            for i in range(classes):
                heat[..., i] = tensor[...] == i
            return heat

        onehot_action = one_hot(action, self.num_of_actions)

        # add to agent's database
        self.observations = np.vstack((self.observations, observation))
        self.actions = np.vstack((self.actions, onehot_action))
        self.rewards = np.vstack((self.rewards, reward))
        self.isdone = np.vstack((self.isdone, isdone))

        # self_train
        if len(self.observations) > 100:
            # self.train(epochs=1)
            pass

    # train agent with some of its collected data from its database
    def train(self, epochs=10):
        qtrain = self.qtrain
        observations, actions, rewards, isdone = self.observations, self.actions, self.rewards, self.isdone
        length = len(observations)

        print('----trainning for', epochs, 'epochs')

        for i in range(epochs):
            # train 1 epoch on a randomly selected subset of the whole database.
            if epochs - 1 == i or i == 0:
                verbose = 2
            else:
                verbose = 0

            subset_size = min(length - 1, 1024)

            indices = np.random.choice(length - 1, subset_size, replace=False)

            subset_observations = np.take(observations, indices, axis=0).astype('float32')
            subset_actions = np.take(actions, indices, axis=0).astype('float32')
            subset_rewards = np.take(rewards, indices, axis=0).astype('float32')
            subset_isdone = np.take(isdone, indices, axis=0).astype('float32')

            subset_next_observations = np.take(observations, indices + 1, axis=0).astype('float32')

            qs2 = self.qfunc.predict(subset_next_observations)

            qtrain.fit([
                subset_observations,
                subset_actions,
                subset_rewards,
                subset_isdone,
                qs2
            ], np.random.rand(subset_size),
                batch_size=64,
                nb_epoch=1,
                verbose=verbose,
                shuffle=False)
        print('----done')


from gym import wrappers
import environments.planeball as planeball
# give it a try
env = planeball.PlaneBallEnv()
# env = gym.make('Acrobot-v1')
#env = gym.make('Pendulum-v0')
# env = gym.make('LunarLander-v2')
# env = gym.make('CartPole-v1')
# env = gym.make('MountainCar-v0')
# env = wrappers.Monitor(env,'./experiment-3',force=True)

if isinstance(env.action_space, Box):
    action_count, d2c = continuous_actions(env)
    num_of_actions = action_count

else:
    num_of_actions = env.action_space.n

num_of_observations = env.observation_space.shape[0]

print('environment:', num_of_actions, 'actions,', num_of_observations, 'observations')

agent = nnagent(
    num_of_actions=num_of_actions,
    num_of_observations=num_of_observations * 2,
    discount_factor=.99,
    epsilon=1.,
    optimizer=RMSprop()
    # optimizer = SGD(lr=0.0005, clipnorm=10.,momentum=0.0,nesterov=False) # momentum must = 0; use plain SGD
)


# main training loop
def r(times=3):
    for k in range(times):
        print('training loop', k, '/', times)
        for i in range(1):  # do 1 episode
            print('play episode:', i)
            do_episode_collect_trajectory(agent, env, max_steps=1000, render=True, feed=True)
            # after play, the episodic data will be feeded to the agent AUTOMATICALLY, so no feeding here

        # wait until collected data became diverse enough
        if len(agent.observations) > 100:
            # ask agent to train itself, with previously collected data
            agent.train(epochs=min(30, len(agent.observations) / 4))

            # decrease epsilon to make agent choose less and less random actions.
            agent.epsilon -= .02
            agent.epsilon = max(0.05, agent.epsilon)
            print('agent epsilon:', agent.epsilon)


def check():
    do_episode_collect_trajectory(agent, env, max_steps=1000, render=True, feed=False, realtime=True, use_best=True)