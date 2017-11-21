import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = np.zeros((1,1,6,3)+((len(actions)),))

    def choose_action(self, observation):

        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            #state_action = self.q_table.ix[observation, :]
            #state_action = state_action.reindex(np.random.permutation(state_action.index))     # some actions have same value
            action = np.argmax(self.q_table[observation])
        else:
            # choose random action
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_):
        best_q = np.amax(self.q_table[s])
        q_predict = self.q_table[s + (a,)]
        q_target = r + self.gamma * best_q  # next state is not terminal
        self.q_table[s + (a,)] += self.lr * (q_target - q_predict)  # update

        return self.q_table

