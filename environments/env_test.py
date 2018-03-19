import logging
import math
from gym import spaces
from gym.utils import seeding
import numpy as np
import gym
import environments.planeball as planeball
import environments.cartpole as cartpole
import pypge

import environments.CartPole3DInterface as c1

env1 = cartpole.CartPoleEnv()

env = planeball.PlaneBallEnv()
env.reset()

'''
for _ in range(1000):
    #env.render()
    env.step(env.action_space.sample()) # take a random action



'''

episode_count = 400
max_steps = 60 * 10
reward = 0
totalReward = 0
done = False

for i in range(episode_count):
    ob = env.reset()
    #env.render()

    for j in range(max_steps):
        action = env.action_space.sample()

        ob, reward, done, _ = env.step(action)

        totalReward += reward

        reward = 0.0

        if done:
            print("Total reward: " + str(totalReward))
            totalReward = 0
            reward = -1.0
            break
            

