#Based on the code of Keon and Morvan

import timeit
import h5py
import random
import bisect
import gym
from gym.spaces import Discrete, Box
import numpy as np
from DQN_Agent import DQNAgent

import environments.planeball as planeball


EPISODES = 1000
ENV_NAME = planeball.PlaneBallEnv()



def cont_to_dis(action):
    action_discrete = []
    for i in range(2):
        section = np.array([-2., -1.5, -1., -0.5, 0, 0.5, 1, 1.5, 2])
        n = bisect.bisect(section, action[i])
        x = -2 + 0.5 * n
        if action[i] - x < 0.25:
            action_dis = x - 0.5
        else:
            action_dis = x
        action_discrete.append(action_dis)

    action_new = np.array(action_discrete)
    #print("new action:")
    #print(action_new)
    return action_new


if __name__ == "__main__":
    env = gym.make(ENV_NAME)
    state_size = env.observation_space.shape[0]

    # determine continuous or discrete action space
    try:
        action_size = env.action_space.shape[0]
        action_bound = env.action_space.high
        # Ensure action bound is symmetric
        assert (env.action_space.high == -env.action_space.low)
        discrete = False
        print('Continuous Action Space')
    except AttributeError:
        action_size = env.action_space.n
        action_bound = 1
        discrete = True
        print('Discrete Action Space')

    agent = DQNAgent(state_size, 81)
    #agent.load("./PlaneBall/save/planeball-dqn.h5")
    #print("Neural Network weights:" + str(agent.get_weight()))
    done = False
    batch_size = 32
    total_steps = 0
    num_streaks = 0

    for e in range(EPISODES):
        start = timeit.default_timer()

        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            #env.render()

            action = agent.act(state)

            if not discrete:
                # Discrete action value
                action = cont_to_dis(action)

            next_state, reward, done, _ = env.step(action)

            #reward = reward if not done else -10


            next_state = np.reshape(next_state, [1, state_size])

            agent.remember(state, action, reward, next_state, done)

            if len(agent.memory) > 1000:
                #print("memory size:", len(agent.memory))
                agent.replay(batch_size)

            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                if (time >= 199):
                    num_streaks += 1
                else:
                    num_streaks = 0
                break
            total_steps += 1

        print("hold 199 in:", num_streaks)

        if num_streaks >= 20:
            env.render()
            print("Find the solution in Episode %d . " % e)
            stop = timeit.default_timer()
            print(stop - start)
            agent.save("./PlaneBall/save/planeball-dqn.h5")
            print("Neural Network weights:" + str(agent.get_weight()))
            break

