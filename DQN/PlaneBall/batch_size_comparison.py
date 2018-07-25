import numpy as np
import gym
import json

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam, SGD

#from DQN.dqn import DQNAgent
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from matplotlib import pyplot
from matplotlib.ticker import FuncFormatter
from keras.models import model_from_json

'''
### batch size comparison: 320, 640, 960, fixed: Boltzman, 0.1, lr=0.01


with open('save/history4_2018-06-22 12:10:41', 'r') as f:
    pp1_1 = json.load(f)
    f.close()

with open('save/history4_2018-06-25 09:14:23', 'r') as f:
    pp1_2 = json.load(f)
    f.close()

with open('save/history4_2018-06-25 13:57:27', 'r') as f:
    pp1_3 = json.load(f)
    f.close()




with open('save/history6_2018-06-22 12:10:47', 'r') as f:
    pp2_1 = json.load(f)
    f.close()

with open('save/history6_2018-06-25 09:14:29', 'r') as f:
    pp2_2 = json.load(f)
    f.close()

with open('save/history6_2018-06-25 16:55:28', 'r') as f:
    pp2_3 = json.load(f)
    f.close()




with open('save/history7_2018-06-22 12:10:49', 'r') as f:
    pp3_1 = json.load(f)
    f.close()

with open('save/history7_2018-06-25 09:14:32', 'r') as f:
    pp3_2 = json.load(f)
    f.close()

with open('save/history7_2018-06-25 16:55:22', 'r') as f:
    pp3_3 = json.load(f)
    f.close()

'''
### batch size comparison: 320, 640, 960, fixed: Boltzman, 0.1, lr=0.1


with open('save/history9_2018-06-26 10:21:24', 'r') as f:
    pp1_1 = json.load(f)
    f.close()

with open('save/history9_2018-06-26 10:21:39', 'r') as f:
    pp1_2 = json.load(f)
    f.close()

with open('save/history9_2018-06-26 10:21:49', 'r') as f:
    pp1_3 = json.load(f)
    f.close()


with open('save/history8_2018-06-25 19:14:36', 'r') as f:
    pp2_1 = json.load(f)
    f.close()

with open('save/history8_2018-06-26 09:17:52', 'r') as f:
    pp2_2 = json.load(f)
    f.close()

with open('save/history8_2018-06-26 09:19:07', 'r') as f:
    pp2_3 = json.load(f)
    f.close()


with open('save/history10_2018-06-27 10:24:29', 'r') as f:
    pp3_1 = json.load(f)
    f.close()

with open('save/history10_2018-06-27 10:24:52', 'r') as f:
    pp3_2 = json.load(f)
    f.close()

with open('save/history10_2018-06-27 10:25:01', 'r') as f:
    pp3_3 = json.load(f)
    f.close()


duration1 = (sum(pp1_1['duration'])+sum(pp1_2['duration'])+sum(pp1_3['duration']))/3
duration2 = (sum(pp2_1['duration'])+sum(pp2_2['duration'])+sum(pp2_3['duration']))/3
duration3 = (sum(pp3_1['duration'])+sum(pp3_2['duration'])+sum(pp3_3['duration']))/3

print('Duration1:{}, Duration2:{}, Duration3:{}'.format(duration1,duration2,duration3))

er_ave1 = [(pp1_1['episode_reward'][i] + pp1_2['episode_reward'][i]+pp1_3['episode_reward'][i])/3 for i in range(len(pp1_1['episode_reward']))]
er_ave2 = [(pp2_1['episode_reward'][i] + pp2_2['episode_reward'][i]+pp2_3['episode_reward'][i])/3 for i in range(len(pp2_1['episode_reward']))]
er_ave3 = [(pp3_1['episode_reward'][i] + pp3_2['episode_reward'][i]+pp3_3['episode_reward'][i])/3 for i in range(len(pp3_1['episode_reward']))]
#er_ave4 = [(pp4_1['episode_reward'][i] + pp4_2['episode_reward'][i]+pp4_3['episode_reward'][i]+pp4_4['episode_reward'][i]+pp4_5['episode_reward'][i])/5 for i in range(len(pp4_1['episode_reward']))]

st_ave1 = [(pp1_1['nb_steps'][i] + pp1_2['nb_steps'][i]+pp1_3['nb_steps'][i])/3 for i in range(len(pp1_1['nb_steps']))]
st_ave2 = [(pp2_1['nb_steps'][i] + pp2_2['nb_steps'][i]+pp2_3['nb_steps'][i])/3 for i in range(len(pp2_1['nb_steps']))]
st_ave3 = [(pp3_1['nb_steps'][i] + pp3_2['nb_steps'][i]+pp3_3['nb_steps'][i])/3 for i in range(len(pp3_1['nb_steps']))]

est_ave1 = [(pp1_1['nb_episode_steps'][i] + pp1_2['nb_episode_steps'][i]+pp1_3['nb_episode_steps'][i])/3 for i in range(len(pp1_1['nb_episode_steps']))]
est_ave2 = [(pp2_1['nb_episode_steps'][i] + pp2_2['nb_episode_steps'][i]+pp2_3['nb_episode_steps'][i])/3 for i in range(len(pp2_1['nb_episode_steps']))]
est_ave3 = [(pp3_1['nb_episode_steps'][i] + pp3_2['nb_episode_steps'][i]+pp3_3['nb_episode_steps'][i])/3 for i in range(len(pp3_1['nb_episode_steps']))]

#pyplot.subplot(2, 1, 1)

pyplot.figure(num=1, figsize=(20, 10),)
pyplot.xlabel('total steps', fontsize=24)
pyplot.ylabel('rewards per episode', fontsize=24)
pyplot.title('batch size comparison-reward', fontsize=24)
new_ticks = np.linspace(0, 2500000, 30)
pyplot.xticks(new_ticks)
pyplot.plot(st_ave1, er_ave1, 'coral', label='batch_size=320')
pyplot.plot(st_ave2, er_ave2, 'mediumseagreen', label='batch_size=640')
pyplot.plot(st_ave3, er_ave3, 'lightskyblue', label='batch_size=960')
pyplot.legend()
pyplot.savefig('save/pics/planeball_batch_size_reward2.png',bbox_inches='tight')



pyplot.figure(num=4, figsize=(20, 10),)
pyplot.xlabel('total episodes', fontsize=24)
pyplot.ylabel('rewards per episode', fontsize=24)
pyplot.title('batch size comparison-reward', fontsize=24)
new_ticks = np.linspace(0, 2500, 30)
pyplot.xticks(new_ticks)
pyplot.plot(er_ave1, 'coral', label='batch_size=320')
pyplot.plot(er_ave2, 'mediumseagreen', label='batch_size=640')
pyplot.plot(er_ave3, 'lightskyblue', label='batch_size=960')
pyplot.legend()
pyplot.savefig('save/pics/planeball_batch_size_reward3.png',bbox_inches='tight')



pyplot.figure(num=2, figsize=(20, 10),)
pyplot.xlabel('total steps', fontsize=24)
pyplot.ylabel('steps per episode', fontsize=24)
pyplot.title('batch size comparison-steps', fontsize=24)
new_ticks = np.linspace(0, 2500000, 30)
pyplot.xticks(new_ticks)
pyplot.plot(st_ave1, est_ave1, 'coral', label='batch_size=320')
pyplot.plot(st_ave2, est_ave2, 'mediumseagreen', label='batch_size=640')
pyplot.plot(st_ave3, est_ave3, 'lightskyblue', label='batch_size=960')
#pyplot.plot(pp1_4['nb_steps'], pp1_4['episode_reward'], 'y', label='640,0.1')
#pyplot.plot(pp1_5['nb_steps'], pp1_5['episode_reward'], 'orange', label='960,0.1')
pyplot.legend()
pyplot.savefig('save/pics/planeball_batch_size_steps2.png',bbox_inches='tight')



pyplot.figure(num=3, figsize=(8, 5),)
width = 0.2
x = np.arange(3)
duration_compare = [duration1,duration2,duration3]
fig, ax = pyplot.subplots()
rects1 = ax.bar(x, duration_compare, width)

pyplot.xticks(x, ('batch_size=320', 'batch_size=640', 'batch_size=960'))
ax.set_ylabel('Duration')
ax.set_title('Training time with different batch size')
pyplot.savefig('save/pics/planeball_duration_BS2.png',bbox_inches='tight')

pyplot.show()
