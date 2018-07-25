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

### target net update interval comparison:0.1, 0.5, 1, 10000, in EpsDisGreedy, 64


with open('save/history9_2018-06-20 13:59:14', 'r') as f:
    pp1_1 = json.load(f)
    f.close()

with open('save/history13_2018-06-22 10:47:37', 'r') as f:
    pp1_2 = json.load(f)
    f.close()

with open('save/history12_2018-06-22 10:44:51', 'r') as f:
    pp1_3 = json.load(f)
    f.close()

with open('save/history14_2018-06-22 10:48:51', 'r') as f:
    pp1_4 = json.load(f)
    f.close()

with open('save/history3_2018-06-20 12:09:36', 'r') as f:
    pp1_5 = json.load(f)
    f.close()


with open('save/history4_2018-06-20 11:09:16', 'r') as f:
    pp2_1 = json.load(f)
    f.close()

with open('save/history4_2018-06-20 12:14:17', 'r') as f:
    pp2_2 = json.load(f)
    f.close()

with open('save/history4_2018-06-20 13:57:53', 'r') as f:
    pp2_3 = json.load(f)
    f.close()

with open('save/history4_2018-06-20 14:57:42', 'r') as f:
    pp2_4 = json.load(f)
    f.close()

with open('save/history4_2018-06-21 09:44:05', 'r') as f:
    pp2_5 = json.load(f)
    f.close()



with open('save/history5_2018-06-20 11:00:11', 'r') as f:
    pp3_1 = json.load(f)
    f.close()

with open('save/history5_2018-06-20 12:14:32', 'r') as f:
    pp3_2 = json.load(f)
    f.close()

with open('save/history5_2018-06-20 13:58:05', 'r') as f:
    pp3_3 = json.load(f)
    f.close()

with open('save/history5_2018-06-20 14:58:05', 'r') as f:
    pp3_4 = json.load(f)
    f.close()

with open('save/history5_2018-06-21 09:44:07', 'r') as f:
    pp3_5 = json.load(f)
    f.close()
'''
duration1 = (sum(pp1_1['duration'])+sum(pp1_2['duration'])+sum(pp1_3['duration'])+sum(pp1_4['duration'])+sum(pp1_5['duration']))/5
duration2 = (sum(pp2_1['duration'])+sum(pp2_2['duration'])+sum(pp2_3['duration'])+sum(pp2_4['duration'])+sum(pp2_5['duration']))/5
duration3 = (sum(pp3_1['duration'])+sum(pp3_2['duration'])+sum(pp3_3['duration'])+sum(pp3_4['duration'])+sum(pp3_5['duration']))/5

print('Duration1:{}, Duration2:{}, Duration3:{}'.format(duration1,duration2,duration3))

st_ave1 = [(pp1_1['nb_steps'][i] + pp1_2['nb_steps'][i]+pp1_3['nb_steps'][i]+pp1_4['nb_steps'][i]+pp1_5['nb_steps'][i])/5 for i in range(len(pp1_1['episode_reward']))]
st_ave2 = [(pp2_1['nb_steps'][i] + pp2_2['nb_steps'][i]+pp2_3['nb_steps'][i]+pp2_4['nb_steps'][i]+pp2_5['nb_steps'][i])/5 for i in range(len(pp2_1['episode_reward']))]
st_ave3 = [(pp3_1['nb_steps'][i] + pp3_2['nb_steps'][i]+pp3_3['nb_steps'][i]+pp3_4['nb_steps'][i]+pp3_5['nb_steps'][i])/5 for i in range(len(pp3_1['episode_reward']))]
#er_ave4 = [(pp4_1['episode_reward'][i] + pp4_2['episode_reward'][i]+pp4_3['episode_reward'][i]+pp4_4['episode_reward'][i]+pp4_5['episode_reward'][i])/5 for i in range(len(pp4_1['episode_reward']))]

er_ave1 = [(pp1_1['episode_reward'][i] + pp1_2['episode_reward'][i]+pp1_3['episode_reward'][i]+pp1_4['episode_reward'][i]+pp1_5['episode_reward'][i])/5 for i in range(len(pp1_1['episode_reward']))]
er_ave2 = [(pp2_1['episode_reward'][i] + pp2_2['episode_reward'][i]+pp2_3['episode_reward'][i]+pp2_4['episode_reward'][i]+pp2_5['episode_reward'][i])/5 for i in range(len(pp2_1['episode_reward']))]
er_ave3 = [(pp3_1['episode_reward'][i] + pp3_2['episode_reward'][i]+pp3_3['episode_reward'][i]+pp3_4['episode_reward'][i]+pp3_5['episode_reward'][i])/5 for i in range(len(pp3_1['episode_reward']))]

'''
#figure 1
pyplot.figure(num=1, figsize=(20, 10),)
pyplot.xlabel('episode')
pyplot.ylabel('rewards per episode')
pyplot.title('Target net update interval')

new_ticks = np.linspace(0, 3000, 30)
pyplot.xticks(new_ticks)
pyplot.plot(pp1_1['episode_reward'], 'r', label='0.1')
pyplot.plot(pp1_2['episode_reward'], 'g', label='0.5')
pyplot.plot(pp1_3['episode_reward'], 'b', label='1.0')
pyplot.plot(pp1_4['episode_reward'], 'y', label='10000')

'''
pyplot.legend()
pyplot.savefig('save/pics/batch_size.png',bbox_inches='tight')


# figure 2
pyplot.figure(num=2, figsize=(8, 5),)
width = 0.2
x = np.arange(3)
duration_compare = [duration1,duration2,duration3]
fig, ax = pyplot.subplots()
rects1 = ax.bar(x, duration_compare, width)

pyplot.xticks(x, ('batch_size=32', 'batch_size=64', 'batch_size=96'))
ax.set_ylabel('Duration')
ax.set_title('Training time with different batch size')
pyplot.savefig('save/pics/duration_BS.png',bbox_inches='tight')
'''
pyplot.show()
