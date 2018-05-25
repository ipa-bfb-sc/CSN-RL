import numpy as np
import math




class QLearningTable:

    def __init__(self, actions, reward_decay=0.99):
        self.actions = actions  # a list
        #self.lr = max(0.01, min(1, 1.0 - math.log10((1)/25)))
        self.gamma = reward_decay
        #self.epsilon = max(0.1, min(0.5, 1.0 - math.log10((1)/25)))
        self.q_table = np.zeros((1, 1, 6, 3)+((len(actions)),))

    def choose_action(self, observation, explore_rate):

        # action selection
        if np.random.uniform() < explore_rate:
            # choose best action
            #state_action = self.q_table.ix[observation, :]
            #state_action = state_action.reindex(np.random.permutation(state_action.index))     # some actions have same value
            action = np.random.choice(self.actions)

        else:
            # choose random action
            action = np.argmax(self.q_table[observation])

        return action

    def learn(self, s, a, r, s_, lr):
        best_q = np.amax(self.q_table[s_])
        q_predict = self.q_table[s + (a,)]
        q_target = r + self.gamma * best_q  # next state is not terminal
        self.q_table[s + (a,)] += lr * (q_target - q_predict)  # update

        return self.q_table

