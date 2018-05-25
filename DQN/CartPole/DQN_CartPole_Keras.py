import random
import math
import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
#when we achieve the maximum limit of out deque,
# it will simply pop out the items from the opposite end


class DQNBrain:
    def __init__(self, observation_size, action_size):
        self.explore_rate = 1
        self.observation_size = observation_size
        self.action_size = action_size
        self.discount_rate = 0.95
        self.learning_rate = 0.1
        self.min_explore_rate = 0.01
        self.decay_explore_rate = 0.995
        self.memory = deque(maxlen=2000)
        self.model = self.build_model()



    def get_explore_rate(self, episode):
        self.explore_rate = max(self.min_explore_rate, min(1, 1.0 - math.log10((episode + 1) / 25)))
        return self.explore_rate

    #def get_learning_rate(self, episode):
     #   self.learning_rate = max(self.min_learning_rate, min(0.5, 1.0 - math.log10((episode + 1) / 25)))
     #   return self.learning_rate

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.observation_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, observation, action, reward, observation_, done):
        self.memory.append((observation,action,reward,observation_,done))

    def choose_action(self, observation):
        #print("explore rate:", self.explore_rate)
        if np.random.rand() <= self.explore_rate:
            action = random.randrange(self.action_size)
            print("random select action:", action)
        else:
            action = np.argmax(self.model.predict(observation))
            print("choose action based on Q decision:", action)
        return action

    def experience_reply(self, num, batch_size=32):
        minibatch = random.sample(self.memory, batch_size)
        for observation, action, reward, observation_, done in minibatch:
            if done:
                Q_target = reward
            else:
                Q_target = reward + self.discount_rate * np.amax(self.model.predict(observation_))

            Q_predict = self.model.predict(observation)
            Q_predict[0][action]= Q_target
            self.model.fit(observation, Q_predict, epochs=1, verbose=0)

        self.explore_rate = self.get_explore_rate(num)
        print("explore rate:", self.explore_rate)

if __name__ == "__main__":

    env = gym.make('CartPole-v0')
    ob_size = env.observation_space.shape[0]
    ac_size = env.action_space.n
    agent = DQNBrain(ob_size, ac_size)
    episodes = 1000

    num_streaks = 0
    total_steps = 0


    for e in range(episodes):
        observation = env.reset()
        observation = np.reshape(observation, [1, ob_size])

        for t in range(500):
            env.render()
            action = agent.choose_action(observation)

            observation_, reward, done, info = env.step(action)

            x, x_dot, theta, theta_dot = observation_
            r1 = (env.observation_space.high[0] - abs(x)) / env.observation_space.high[0] - 0.8
            r2 = (env.observation_space.high[2] - abs(theta)) / env.observation_space.high[2] - 0.5
            reward = r1 + r2
            #print(observation_)
            observation_ = np.reshape(observation_, [1, ob_size])
            #print(observation_)

            agent.remember(observation, action, reward, observation_, done)
            if len(agent.memory) > batch_size:
                agent.experience_reply(e)

            observation = observation_

            if done:
                print("Episode %d finished after %f time steps" % (e, t))
                if (t >= 199):
                    num_streaks += 1
                else:
                    num_streaks = 0
                break

            #total_steps += 1

        print("hold 199 in:", num_streaks)


            #if total_steps > 100:
        #    agent.experience_reply(e)

        if num_streaks > 120:
            print("Find the solution in Episode %d . " % episode)
            break











