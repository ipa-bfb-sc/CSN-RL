import numpy as np
import gym
import json



from matplotlib import pyplot
from matplotlib.ticker import FuncFormatter
from keras.models import model_from_json


### batch size comparison: 32, 64, 96, fixed: Boltzman0.8, 0.1, lr=0.01


with open('save/history5_2018-07-04 14:29:38', 'r') as f:
    pp1_1 = json.load(f)
    f.close()

with open('save/history5_2018-07-04 14:29:54', 'r') as f:
    pp1_2 = json.load(f)
    f.close()

with open('save/history5_2018-07-04 14:30:10', 'r') as f:
    pp1_3 = json.load(f)
    f.close()


with open('save/history3_2018-07-04 10:28:40', 'r') as f:
    pp2_1 = json.load(f)
    f.close()

with open('save/history3_2018-07-04 10:30:46', 'r') as f:
    pp2_2 = json.load(f)
    f.close()

with open('save/history3_2018-07-04 10:31:26', 'r') as f:
    pp2_3 = json.load(f)
    f.close()


with open('save/history4_2018-07-04 10:17:11', 'r') as f:
    pp3_1 = json.load(f)
    f.close()

with open('save/history4_2018-07-04 10:17:45', 'r') as f:
    pp3_2 = json.load(f)
    f.close()

with open('save/history4_2018-07-04 10:23:04', 'r') as f:
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
new_ticks = np.linspace(0, 1200000, 30)
new_ticksy = np.linspace(-200, 1500, 18)

pyplot.xticks(new_ticks)
pyplot.yticks(new_ticksy)

pyplot.plot(st_ave1, er_ave1, 'hotpink', label='batch_size=32')
pyplot.plot(st_ave2, er_ave2, 'palegreen', label='batch_size=64')
pyplot.plot(st_ave3, er_ave3, 'deepskyblue', label='batch_size=96')
pyplot.legend()
pyplot.savefig('save/pics/cirturtle_batch_size_reward2.png',bbox_inches='tight')



pyplot.figure(num=4, figsize=(20, 10),)
pyplot.xlabel('total episodes', fontsize=24)
pyplot.ylabel('rewards per episode', fontsize=24)
pyplot.title('batch size comparison-reward', fontsize=24)
new_ticks = np.linspace(0, 2500, 30)
new_ticksy = np.linspace(-200, 1500, 18)

pyplot.xticks(new_ticks)
pyplot.yticks(new_ticksy)
pyplot.plot(er_ave1, 'hotpink', label='batch_size=32')
pyplot.plot(er_ave2, 'palegreen', label='batch_size=64')
pyplot.plot(er_ave3, 'deepskyblue', label='batch_size=96')
pyplot.legend()
pyplot.savefig('save/pics/cirturtle_batch_size_reward.png',bbox_inches='tight')



pyplot.figure(num=2, figsize=(20, 10),)
pyplot.xlabel('total steps', fontsize=24)
pyplot.ylabel('steps per episode', fontsize=24)
pyplot.title('batch size comparison-steps', fontsize=24)
new_ticks = np.linspace(0, 1200000, 30)

pyplot.xticks(new_ticks)

pyplot.plot(st_ave1, est_ave1, 'hotpink', label='batch_size=32')
pyplot.plot(st_ave2, est_ave2, 'palegreen', label='batch_size=64')
pyplot.plot(st_ave3, est_ave3, 'deepskyblue', label='batch_size=96')
#pyplot.plot(pp1_4['nb_steps'], pp1_4['episode_reward'], 'y', label='640,0.1')
#pyplot.plot(pp1_5['nb_steps'], pp1_5['episode_reward'], 'orange', label='960,0.1')
pyplot.legend()
pyplot.savefig('save/pics/cirturtle_batch_size_steps.png',bbox_inches='tight')



pyplot.figure(num=3, figsize=(8, 5),)
width = 0.2
x = np.arange(3)
duration_compare = [duration1,duration2,duration3]
fig, ax = pyplot.subplots()
rects1 = ax.bar(x, duration_compare, width)

pyplot.xticks(x, ('batch_size=32', 'batch_size=64', 'batch_size=96'))
ax.set_ylabel('Duration')
ax.set_title('Training time with different batch size')
pyplot.savefig('save/pics/cirturtle_duration_BS.png',bbox_inches='tight')

pyplot.show()
