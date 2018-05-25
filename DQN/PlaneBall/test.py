
from gym.spaces import Discrete, Box
import numpy as np
import environments.planeball as planeball

import bisect


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
            idx = idx / split
            idx.astype(int)
            continuous.append(rem)
            print("rem:")
            print(rem)

        cont = (np.array(continuous, dtype='float32') / (split - 1)) * itvl + low
        return cont

    return action_count, d2c

def cont_to_dis(action):
    action_discrete = []
    for i in range(2):
        section = np.array([-2., -1.5, -1., -0.5, 0, 0.5, 1, 1.5, 2])
        n = bisect.bisect(section, action[i])
        x = -2 + 0.5*n
        if action[i] - x < 0.25:
            action_dis = x - 0.5
        else:
            action_dis = x
        action_discrete.append(action_dis)

    action_new = np.array(action_discrete)
    #print("new action:")
    #print(action_new)
    return action_new


def action_index(self, one_action):

    bb = np.rec.fromarrays(one_action) #
    x, y = np.meshgrid(np.linspace(-2., 2., 9), np.linspace(-2., 2., 9))
    ac = np.rec.fromarrays([y, x])
    ac2 = ac.flatten()
    p = np.where(ac2 == bb)
    num = p[0]
    return num

def one_hot(action, num):
    heat = np.zeros(action.shape + (num,))
    print("heat:")
    print(heat)
    for i in range(num):
        heat[..., i] = action[...] == i
    return heat



if __name__ == "__main__":
    #print(np.random.uniform(-2,2,(1,2)))
    #print(np.random.uniform(-2, 2, (2)))
    env = planeball.PlaneBallEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    print(state_size,action_size)

    state = env.reset()
    print(state)
    state = np.reshape(state, [1, state_size])
    print(state)

    for time in range(5):
        #env.render()

        action = env.action_space.sample()
        print("0:")
        print(action)
        print(action.shape)
        print(action[0])
        print(action[1])

        aa = cont_to_dis(action)
        print("1:")
        print(aa)

        #bb = np.reshape(aa, [1, 2])
        bb = np.rec.fromarrays(aa)
        print(bb)



        #onehot_actions = one_hot(action, 81)
        print("2:")
        #print(onehot_actions)

        x, y = np.meshgrid(np.linspace(-2., 2., 9), np.linspace(-2., 2., 9))
        ac = np.rec.fromarrays([y, x])
        ac2 = ac.flatten()
        #ac2 = np.reshape(ac, [1, 81])

        #nn = bisect.bisect(ac2, bb)

        cc = list(ac2[2])
        #p = np.where(ac2==bb)
        #num = p[0]
        print(ac)
        print("ac:")
        print(ac2)
        print("ac2:")
        #print(int(num))
        print(ac2[2])
        print(cc[0])


        #action = np.reshape(action, [1, action_size])
        #print("1:")
        #print(action)

        # Discrete action value
'''
        if isinstance(env.action_space, Box):
            action_count, d2c = continuous_actions(env)
            print("12345")
            print(action_count,d2c)
            actual_action = d2c(action)
            print("12345678")
            print(actual_action)
        else:
            actual_action = action
        print("2:")
        print(actual_action)


        #next_state, reward, done, _ = env.step(actual_action)
        '''