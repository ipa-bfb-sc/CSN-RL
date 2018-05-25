
import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class DQNAgent:
    def __init__(self, state_size, action_size, action_bound, action_num = 0, discrete = True):
        self.state_size = state_size
        self.action_size = action_size
        self.action_bound = action_bound
        self.action_num = action_num
        self.discrete = discrete
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001 # learning rate
        self.model = self._build_model()

    # Neural Net for Deep-Q learning Model
    def _build_model(self):

        if self.discrete:
            self.action_layer = self.action_size
        else:
            self.action_layer = self.action_num

        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_layer, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def act(self, state):
        if np.random.rand() <= self.epsilon:
            if self.discrete:
                return random.randrange(self.action_size)
            else:
                print('random:')
                return np.random.uniform(-self.action_bound, self.action_bound, (self.action_size))
        act_values = self.model.predict(state)
        #return np.argmax(act_values[0])  # returns action
        print('choose from Q_table:')
        print(act_values)
        print(np.argmax(act_values))
        return np.argmax(act_values)

    def replay(self, batch_size):

        # create index for actions
        def action_index(one_action):
            if not self.discrete:
                bb = np.rec.fromarrays(one_action)
                x, y = np.meshgrid(np.linspace(-2., 2., 9), np.linspace(-2., 2., 9))
                ac = np.rec.fromarrays([y, x])
                ac2 = ac.flatten()
                p = np.where(ac2 == bb)
                num = p[0]
                return num
            else:
                return one_action
        #########

        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))

            predict = self.model.predict(state)
            print("Q_predict:",predict)

            ind = action_index(action)
            predict[0][ind] = target
            print("Q_target", target)
            print("action_num:", ind)
            print("Q_predict2:", predict)

            self.model.fit(state, predict, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def get_weight(self):
        return self.model.get_weights()
    def get_param(self):
        return self.gamma, self.learning_rate, self.epsilon, self.epsilon_decay

