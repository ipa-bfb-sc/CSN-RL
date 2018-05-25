#!/usr/bin/env python
import gym
import gym_gazebo
import time
import numpy
import random
import time
import timeit
import functools
from DQN_Agent import DQNAgent

import matplotlib
import matplotlib.pyplot as plt



'''
count = q.count(maxQ)
# In case there're several state-action max values
# we select a random one among them

if count > 1:
    best = [i for i in range(len(self.actions)) if q[i] == maxQ]
    i = random.choice(best)
else:
    i = q.index(maxQ)
    
'''

class LivePlot(object):
    def __init__(self, outdir, data_key='episode_rewards', line_color='blue'):
        """
        Liveplot renders a graph of either episode_rewards or episode_lengths
        Args:
            outdir (outdir): Monitor output file location used to populate the graph
            data_key (Optional[str]): The key in the json to graph (episode_rewards or episode_lengths).
            line_color (Optional[dict]): Color of the plot.
        """
        self.outdir = outdir
        self._last_data = None
        self.data_key = data_key
        self.line_color = line_color

        #styling options
        matplotlib.rcParams['toolbar'] = 'None'
        plt.style.use('ggplot')
        plt.xlabel("")
        plt.ylabel(data_key)
        fig = plt.gcf().canvas.set_window_title('simulation_graph')

    def plot(self):
        results = gym.monitoring.monitor.load_results(self.outdir)
        data =  results[self.data_key]

        #only update plot if data is different (plot calls are expensive)
        if data !=  self._last_data:
            self._last_data = data
            plt.plot(data, color=self.line_color)

            # pause so matplotlib will display
            # may want to figure out matplotlib animation or use a different library in the future
            plt.pause(0.000001)

def render():
    render_skip = 0 #Skip first X episodes.
    render_interval = 50 #Show render Every Y episodes.
    render_episodes = 10 #Show Z episodes every rendering.

    if (e%render_interval == 0) and (e != 0) and (e > render_skip):
        env.render()
    elif ((e-render_episodes)%render_interval == 0) and (e != 0) and (e > render_skip) and (render_episodes < e):
        env.render(close=True)


EPISODES = 1000
ENV_NAME = 'GazeboCircuitTurtlebotLidar-v0'

if __name__ == '__main__':

    env = gym.make(ENV_NAME)


    outdir = '~/Thesis-RL/DQN/CirTurtleBot'
    # env.monitor.start(outdir, force=True, seed=None)

    #plotter = LivePlot(outdir)

    last_time_steps = numpy.ndarray(0)

    state_size = env.observation_space.shape[0]

    # determine action space is continuous or discrete
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

    agent = DQNAgent(state_size, action_size, action_bound, discrete = discrete)
    # agent.load("./CartPole/save/cartpole-dqn.h5")
    # print("Neural Network weights:" + str(agent.get_weight()))
    start_time = time.time()
    highest_reward = 0
    done = False
    batch_size = 32
    total_steps = 0
    num_streaks = 0

    for e in range(EPISODES):
        done = False

        cumulated_reward = 0 #Should going forward give more reward then L/R ?

        observation = env.reset()

        render() #defined above, not env.render()

        state = ''.join(map(str, observation))

        for i in range(500):

            # Pick an action based on the current state
            action = agent.act(state)

            # Execute the action and get feedback
            observation, reward, done, info = env.step(action)
            cumulated_reward += reward

            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            next_state = ''.join(map(str, observation))

            agent.remember(state, action, reward, next_state, done)
            if len(agent.memory) > 50:
                print("memory :", agent.memory)
                agent.replay(batch_size)

            # env.monitor.flush(force=True)

            if not(done):
                state = next_state
            else:
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break

        #The divmod() method takes two numbers and returns a pair of numbers (a tuple) consisting of their quotient and remainder.
        # Get hour, minute, second when an episode is done.
        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        print ("EP: "+str(e+1) + " - epsilon: "+str(round(agent.epsilon,2))+" - Reward: "+str(cumulated_reward)+"     Time: %d:%02d:%02d" % (h, m, s))

    #Github table content
    gama, lr, ep, ep_discount = agent.get_param
    print ("\n|"+str(EPISODES)+"|"+str(lr)+"|"+str(gama)+"|"+str(ep)+"*"+str(ep_discount)+"|"+str(highest_reward)+"| PICTURE |")

    l = last_time_steps.tolist()
    l.sort()

    #print("Parameters: a="+str)
    print("Overall score: {:0.2f}".format(last_time_steps.mean()))
    print("Best 100 score: {:0.2f}".format(functools.reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    #env.monitor.close()
    env.close()

