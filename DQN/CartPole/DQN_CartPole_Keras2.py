#Based on the code of Keon and Morvan

import timeit
import h5py

import gym
import numpy as np
from DQN_Agent import DQNAgent


EPISODES = 1000
ENV_NAME = 'CartPole-v0'


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

    agent = DQNAgent(state_size, action_size)
    #agent.load("./save/cartpole-dqn.h5")
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
            next_state, reward, done, _ = env.step(action)
            #reward = reward if not done else -10
            x, x_dot, theta, theta_dot = next_state
            r1 = (env.observation_space.high[0] - abs(x)) / env.observation_space.high[0] - 0.8
            r2 = (env.observation_space.high[2] - abs(theta)) / env.observation_space.high[2] - 0.5
            reward = r1 + r2
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            if len(agent.memory) > 1000:
                #print("memory size:", len(agent.memory))
                agent.replay(batch_size)

            state = next_state
            if done: ## is done: 1.pole angleis more than 12degree. 2.Cart position is more than 2.4 . 3. episode length is greater than 200
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
            agent.save("./save/cartpole-dqn.h5")
            print("Neural Network weights:" + str(agent.get_weight()))
            break

