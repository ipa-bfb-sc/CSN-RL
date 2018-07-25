import numpy as np
import gym
import json


from scipy.interpolate import interp1d
from matplotlib import pyplot
from matplotlib.ticker import FuncFormatter
from keras.models import model_from_json


### memory comparison: 10000, 50000,

# history 1: actor_lr=0.00001, critic_lr=0.001, batch_size=96, memory=50000
with open('save/history1_2018-07-12 18:41:44', 'r') as f:
    pp1_1 = json.load(f)
    f.close()
with open('save/history1_2018-07-12 18:59:06', 'r') as f:
    pp1_2 = json.load(f)
    f.close()
with open('save/history1_2018-07-12 18:59:18', 'r') as f:
    pp1_3 = json.load(f)
    f.close()
with open('save/history1_2018-07-12 18:59:32', 'r') as f:
    pp1_4 = json.load(f)
    f.close()
with open('save/history1_2018-07-12 18:59:37', 'r') as f:
    pp1_5 = json.load(f)
    f.close()


#history3: actor_lr=0.00001, critic_lr=0.001, batch_size=96, memory=10000
with open('save/history3_2018-07-17 13:58:42', 'r') as f:
    pp2_1 = json.load(f)
    f.close()

with open('save/history3_2018-07-17 13:58:42', 'r') as f:
    pp2_2 = json.load(f)
    f.close()

with open('save/history3_2018-07-17 13:59:29', 'r') as f:
    pp2_3 = json.load(f)
    f.close()

with open('save/history3_2018-07-17 15:28:27', 'r') as f:
    pp2_4 = json.load(f)
    f.close()

with open('save/history3_2018-07-17 15:28:35', 'r') as f:
    pp2_5 = json.load(f)
    f.close()


'''

with open('save/history4_2018-07-04 10:17:11', 'r') as f:
    pp3_1 = json.load(f)
    f.close()

with open('save/history4_2018-07-04 10:17:45', 'r') as f:
    pp3_2 = json.load(f)
    f.close()

with open('save/history4_2018-07-04 10:23:04', 'r') as f:
    pp3_3 = json.load(f)
    f.close()


a_step = pp1_1['nb_steps']
a_epr = pp1_1['episode_reward']


# end_training index:
index = []
index.extend((np.where(pp1_1['train_end_episode'])[0][0], np.where(pp1_2['train_end_episode'])[0][0], np.where(pp1_3['train_end_episode'])[0][0], np.where(pp1_4['train_end_episode'])[0][0], np.where(pp1_5['train_end_episode'])[0][0]))

# Get training steps and the average:
train_step = []
train_step.extend((pp1_1['nb_steps'][index[0]], pp1_2['nb_steps'][index[1]], pp1_3['nb_steps'][index[2]], pp1_4['nb_steps'][index[3]], pp1_5['nb_steps'][index[4]]))

ave = np.mean(train_step)
print('End training step:{}, Average steps:{}'.format(train_step, ave))

# Get episode_reward:
episode_reward1 = []
episode_reward2 = []
episode_reward3 = []
episode_reward4 = []
episode_reward5 = []
ep_reward = {}
ep_reward.update({'e1':episode_reward1})
ep_reward.update({'e2':episode_reward2})
ep_reward.update({'e3':episode_reward3})
ep_reward.update({'e4':episode_reward4})
ep_reward.update({'e5':episode_reward5})
print(ep_reward)

for i in range(0,index[0]):
    episode_reward1.append(pp1_1['episode_reward'][i])

for i in range(0,index[1]):
    episode_reward2.append(pp1_2['episode_reward'][i])

for i in range(0,index[2]):
    episode_reward2.append(pp1_3['episode_reward'][i])

for i in range(0,index[3]):
    episode_reward2.append(pp1_4['episode_reward'][i])

for i in range(0,index[4]):
    episode_reward2.append(pp1_5['episode_reward'][i])

print(len(ep_reward['e2']))
'''



duration1 = (sum(pp1_1['duration'])+sum(pp1_2['duration'])+sum(pp1_3['duration'])+sum(pp1_4['duration'])+sum(pp1_5['duration']))/5
duration2 = (sum(pp2_1['duration'])+sum(pp2_2['duration'])+sum(pp2_3['duration'])+sum(pp2_4['duration'])+sum(pp2_5['duration']))/5
#duration3 = (sum(pp3_1['duration'])+sum(pp3_2['duration'])+sum(pp3_3['duration']))/3

#print('Duration1:{}, Duration2:{}'.format(duration1,duration2))

#er_ave1 = [(pp1_1['episode_reward'][i] + pp1_2['episode_reward'][i]+pp1_3['episode_reward'][i])/3 for i in range(len(pp1_1['episode_reward']))]
#er_ave2 = [(pp2_1['episode_reward'][i] + pp2_2['episode_reward'][i]+pp2_3['episode_reward'][i])/3 for i in range(len(pp2_1['episode_reward']))]
#er_ave3 = [(pp3_1['episode_reward'][i] + pp3_2['episode_reward'][i]+pp3_3['episode_reward'][i])/3 for i in range(len(pp3_1['episode_reward']))]
#er_ave4 = [(pp4_1['episode_reward'][i] + pp4_2['episode_reward'][i]+pp4_3['episode_reward'][i]+pp4_4['episode_reward'][i]+pp4_5['episode_reward'][i])/5 for i in range(len(pp4_1['episode_reward']))]

#st_ave1 = [(pp1_1['nb_steps'][i] + pp1_2['nb_steps'][i]+pp1_3['nb_steps'][i])/3 for i in range(len(pp1_1['nb_steps']))]
#st_ave2 = [(pp2_1['nb_steps'][i] + pp2_2['nb_steps'][i]+pp2_3['nb_steps'][i])/3 for i in range(len(pp2_1['nb_steps']))]
#st_ave3 = [(pp3_1['nb_steps'][i] + pp3_2['nb_steps'][i]+pp3_3['nb_steps'][i])/3 for i in range(len(pp3_1['nb_steps']))]

#est_ave1 = [(pp1_1['nb_episode_steps'][i] + pp1_2['nb_episode_steps'][i]+pp1_3['nb_episode_steps'][i])/3 for i in range(len(pp1_1['nb_episode_steps']))]
#est_ave2 = [(pp2_1['nb_episode_steps'][i] + pp2_2['nb_episode_steps'][i]+pp2_3['nb_episode_steps'][i])/3 for i in range(len(pp2_1['nb_episode_steps']))]
#est_ave3 = [(pp3_1['nb_episode_steps'][i] + pp3_2['nb_episode_steps'][i]+pp3_3['nb_episode_steps'][i])/3 for i in range(len(pp3_1['nb_episode_steps']))]

#pyplot.subplot(2, 1, 1)


x = np.linspace(200, 299800, num=300000)

f1_1 = (interp1d(pp1_1['nb_steps'], pp1_1['episode_reward'],'cubic'))(x)
f1_2 = (interp1d(pp1_2['nb_steps'], pp1_2['episode_reward'],'cubic'))(x)
f1_3 = (interp1d(pp1_3['nb_steps'], pp1_3['episode_reward'],'cubic'))(x)
f1_4 = (interp1d(pp1_4['nb_steps'], pp1_4['episode_reward'],'cubic'))(x)
f1_5 = (interp1d(pp1_5['nb_steps'], pp1_5['episode_reward'],'cubic'))(x)

exp1 = np.vstack((f1_1, f1_2, f1_3, f1_4, f1_5))

f_avg1 = np.average(exp1, axis=0)

f2_1 = (interp1d(pp2_1['nb_steps'], pp2_1['episode_reward'],'cubic'))(x)
f2_2 = (interp1d(pp2_2['nb_steps'], pp2_2['episode_reward'],'cubic'))(x)
f2_3 = (interp1d(pp2_3['nb_steps'], pp2_3['episode_reward'],'cubic'))(x)
f2_4 = (interp1d(pp2_4['nb_steps'], pp2_4['episode_reward'],'cubic'))(x)
f2_5 = (interp1d(pp2_5['nb_steps'], pp2_5['episode_reward'],'cubic'))(x)

exp2 = np.vstack((f2_1, f2_2, f2_3, f2_4, f2_5))

f_avg2 = np.average(exp2, axis=0)




pyplot.figure(num=1, figsize=(20, 10),)
pyplot.xlabel('total steps', fontsize=24)
pyplot.ylabel('rewards per episode', fontsize=24)
pyplot.title('memory size comparison-reward', fontsize=24)

#pyplot.plot(f1_int, 'hotpink', label='actor_lr=0.00001')
pyplot.plot(f_avg2, 'green', label='memory=10000')
#pyplot.plot(f5_int, 'blue', label='learn_rate=0.001')
pyplot.plot(f_avg1, 'red', label='memory=50000')
#pyplot.plot(st_ave3, er_ave3, 'deepskyblue', label='batch_size=96')
pyplot.legend()
pyplot.savefig('save/pics/ddpg_cartpole_ms_reward.png',bbox_inches='tight')


pyplot.figure(num=2, figsize=(8, 5),)
width = 0.2
x = np.arange(2)
duration_compare = [duration1,duration2]
fig, ax = pyplot.subplots()
rects1 = ax.bar(x, duration_compare, width)

pyplot.xticks(x, ('memory=50000', 'memory=10000'))
ax.set_ylabel('Duration')
ax.set_title('Training time with different memory size')
pyplot.savefig('save/pics/ddpg_cartpole_duration_MS.png',bbox_inches='tight')

pyplot.show()


'''
pyplot.figure(num=4, figsize=(20, 10),)
pyplot.xlabel('total episodes', fontsize=24)
pyplot.ylabel('rewards per episode', fontsize=24)
pyplot.title('batch size comparison-reward', fontsize=24)
new_ticks = np.linspace(0, 3000, 30)
new_ticksy = np.linspace(-200, 1500, 18)

pyplot.xticks(new_ticks)
pyplot.yticks(new_ticksy)
pyplot.plot(er_ave1, 'hotpink', label='learn_rate=0.01')
pyplot.plot(er_ave2, 'palegreen', label='learn_rate=0.001')
#pyplot.plot(er_ave3, 'deepskyblue', label='batch_size=96')
pyplot.legend()
pyplot.savefig('save/pics/cirturtle_learn_rate_reward.png',bbox_inches='tight')



pyplot.figure(num=2, figsize=(20, 10),)
pyplot.xlabel('total steps', fontsize=24)
pyplot.ylabel('steps per episode', fontsize=24)
pyplot.title('batch size comparison-steps', fontsize=24)
new_ticks = np.linspace(0, 1200000, 30)

pyplot.xticks(new_ticks)

pyplot.plot(st_ave1, est_ave1, 'hotpink', label='learn_rate=0.01')
pyplot.plot(st_ave2, est_ave2, 'palegreen', label='learn_rate=0.001')
#pyplot.plot(st_ave3, est_ave3, 'deepskyblue', label='batch_size=96')
#pyplot.plot(pp1_4['nb_steps'], pp1_4['episode_reward'], 'y', label='640,0.1')
#pyplot.plot(pp1_5['nb_steps'], pp1_5['episode_reward'], 'orange', label='960,0.1')
pyplot.legend()
pyplot.savefig('save/pics/cirturtle_learn_rate_steps.png',bbox_inches='tight')
'''



