import gym
import numpy as np

from keras.models import Model
from keras.layers import *
from keras import backend as K

from collections import deque


def one_hot(index, categories):
    x = np.zeros((categories,))
    x[index] = 1
    return x


def discount_rewards(r, gamma=0.99):
    """ Take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def pg_loss(advantage):
    def f(y_true, y_pred):
        """
        Policy gradient loss
        """
        # L = \sum{A * log(p(y | x))}
        # Mask out probability of action taken
        responsible_outputs = K.sum(y_true * y_pred, axis=1)
        policy_loss = -K.sum(advantage * K.log(responsible_outputs))
        return policy_loss

    return f


def create_model():
    """
    Model architecture
    """
    state = Input(shape=(4,))
    x = Dense(64)(state)
    x = Dense(64)(x)
    x = Dense(2)(x)
    x = Activation('softmax')(x)

    model = Model(state, x)
    return model


def pg(model):
    """
    Wraps the model in a policy gradient model
    """
    state = Input(shape=(4,))
    # Advantages for loss function
    adv_input = Input(shape=(1,))

    x = model(state)

    model = Model([state, adv_input], x)
    model.compile(
        optimizer='nadam',
        loss=pg_loss(adv_input)
    )

    return model


# Create env
env = gym.make('CartPole-v0')

g_model = create_model()
pg_model = pg(g_model)
all_rewards = deque(maxlen=100)

for i_episode in range(10000):
    observation = env.reset()

    # History of this episode
    state_history = []
    action_history = []
    reward_history = []

    for t in range(100):
        # env.render()

        state_history.append(observation)

        action_prob = g_model.predict(np.expand_dims(observation, axis=0))[0]
        action = np.random.choice(len(action_prob), 1, p=action_prob)[0]
        observation, reward, done, info = env.step(action)

        reward_history.append(reward)
        action_history.append(one_hot(action, 2))

        if done:
            reward_sum = sum(reward_history)
            all_rewards.append(reward_sum)

            adv = discount_rewards(reward_history)

            state_history = np.array(state_history)
            action_history = np.array(action_history)

            pg_model.train_on_batch([state_history, adv], action_history)

            print("Episode finished with reward {} {:.2f}".format(reward_sum, np.mean(all_rewards)))
            break