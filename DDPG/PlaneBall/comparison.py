import numpy as np
import gym
import json


from scipy.interpolate import interp1d
from matplotlib import pyplot
from matplotlib.ticker import FuncFormatter
from keras.models import model_from_json


### actor_lr comparison: 0.00001, 0.001, critic_lr comparison: 0.001, 0.01, target_net comparison: 0.001, 0.01

# history 1: actor_lr=0.00001, critic_lr=0.001, batch_size=96, memory=50000, target_update=0.001, discount_rate=0.99
with open('save/history1_2018-07-17 23:11:25', 'r') as f:
    pp1_1 = json.load(f)
    f.close()
with open('save/history1_2018-07-18 00:11:06', 'r') as f:
    pp1_2 = json.load(f)
    f.close()
with open('save/history1_2018-07-18 00:11:13', 'r') as f:
    pp1_3 = json.load(f)
    f.close()
#with open('save/history1_2018-07-12 18:59:32', 'r') as f:
#    pp1_4 = json.load(f)
#    f.close()
#with open('save/history1_2018-07-12 18:59:37', 'r') as f:
#    pp1_5 = json.load(f)
#    f.close()


# history 2: actor_lr=0.001, critic_lr=0.001, batch_size=96, memory=50000, target_update=0.001, discount_rate=0.99

with open('save/history2_2018-07-18 00:18:44', 'r') as f:
    pp2_1 = json.load(f)
    f.close()

with open('save/history2_2018-07-18 00:19:32', 'r') as f:
    pp2_2 = json.load(f)
    f.close()

with open('save/history2_2018-07-18 00:19:50', 'r') as f:
    pp2_3 = json.load(f)
    f.close()


#history3: actor_lr=0.00001, critic_lr=0.001, batch_size=96, memory=200000, target_update=0.001, discount_rate=0.99
with open('save/history3_2018-07-18 00:29:31', 'r') as f:
    pp3_1 = json.load(f)
    f.close()

with open('save/history3_2018-07-18 00:29:38', 'r') as f:
    pp3_2 = json.load(f)
    f.close()

with open('save/history3_2018-07-18 00:29:49', 'r') as f:
    pp3_3 = json.load(f)
    f.close()

#with open('save/history3_2018-07-17 15:28:27', 'r') as f:
#    pp3_4 = json.load(f)
#    f.close()

#with open('save/history3_2018-07-17 15:28:35', 'r') as f:
#    pp3_5 = json.load(f)
#    f.close()


#history4: actor_lr=0.00001, critic_lr=0.01, batch_size=96, memory=50000, target_update=0.001, discount_rate=0.99
with open('save/history4_2018-07-18 09:40:22', 'r') as f:
    pp4_1 = json.load(f)
    f.close()

with open('save/history4_2018-07-18 09:40:42', 'r') as f:
    pp4_2 = json.load(f)
    f.close()

with open('save/history4_2018-07-18 09:40:59', 'r') as f:
    pp4_3 = json.load(f)
    f.close()

#history5: actor_lr=0.00001, critic_lr=0.001, batch_size=96, memory=50000, target_update=0.01, discount_rate=0.99
with open('save/history5_2018-07-18 09:46:37', 'r') as f:
    pp5_1 = json.load(f)
    f.close()

with open('save/history5_2018-07-18 10:52:13', 'r') as f:
    pp5_2 = json.load(f)
    f.close()

with open('save/history5_2018-07-18 09:47:13', 'r') as f:
    pp5_3 = json.load(f)
    f.close()


#history6: actor_lr=0.00001, critic_lr=0.001, batch_size=96, memory=50000, target_update=0.001, discount_rate=0.5
with open('save/history6_2018-07-18 11:30:16', 'r') as f:
    pp6_1 = json.load(f)
    f.close()

with open('save/history6_2018-07-18 11:30:33', 'r') as f:
    pp6_2 = json.load(f)
    f.close()

with open('save/history6_2018-07-18 11:30:48', 'r') as f:
    pp6_3 = json.load(f)
    f.close()


duration1 = (sum(pp1_1['duration'])+sum(pp1_2['duration'])+sum(pp1_3['duration']))/3
duration2 = (sum(pp2_1['duration'])+sum(pp2_2['duration'])+sum(pp2_3['duration']))/3
duration3 = (sum(pp3_1['duration'])+sum(pp3_2['duration'])+sum(pp3_3['duration']))/3
duration4 = (sum(pp4_1['duration'])+sum(pp4_2['duration'])+sum(pp4_3['duration']))/3
duration5 = (sum(pp5_1['duration'])+sum(pp5_2['duration'])+sum(pp5_3['duration']))/3

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


reward1 = (sum(pp1_1['episode_reward'])+sum(pp1_2['episode_reward'])+sum(pp1_3['episode_reward']))/3
reward2 = (sum(pp4_1['episode_reward'])+sum(pp4_2['episode_reward'])+sum(pp4_3['episode_reward']))/3
print(reward1, reward2)



  ### kind: 'linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic, ‘cubic’
x = np.linspace(1000, 499000, num=500000)

f1_1 = (interp1d(pp1_1['nb_steps'], pp1_1['episode_reward']))(x)
f1_2 = (interp1d(pp1_2['nb_steps'], pp1_2['episode_reward']))(x)
f1_3 = (interp1d(pp1_3['nb_steps'], pp1_3['episode_reward']))(x)
exp1 = np.vstack((f1_1, f1_2, f1_3))
f_avg1 = np.average(exp1, axis=0)



f2_1 = (interp1d(pp2_1['nb_steps'], pp2_1['episode_reward']))(x)
f2_2 = (interp1d(pp2_2['nb_steps'], pp2_2['episode_reward']))(x)
f2_3 = (interp1d(pp2_3['nb_steps'], pp2_3['episode_reward']))(x)
exp2 = np.vstack((f2_1, f2_2, f2_3))
f_avg2 = np.average(exp2, axis=0)



f3_1 = (interp1d(pp3_1['nb_steps'], pp3_1['episode_reward']))(x)
f3_2 = (interp1d(pp3_2['nb_steps'], pp3_2['episode_reward']))(x)
f3_3 = (interp1d(pp3_3['nb_steps'], pp3_3['episode_reward']))(x)
exp3 = np.vstack((f3_1, f3_2, f3_3))
f_avg3 = np.average(exp3, axis=0)


f4_1 = (interp1d(pp4_1['nb_steps'], pp4_1['episode_reward']))(x)
f4_2 = (interp1d(pp4_2['nb_steps'], pp4_2['episode_reward']))(x)
f4_3 = (interp1d(pp4_3['nb_steps'], pp4_3['episode_reward']))(x)
exp4 = np.vstack((f4_1, f4_2, f4_3))
f_avg4 = np.average(exp4, axis=0)


f5_1 = (interp1d(pp5_1['nb_steps'], pp5_1['episode_reward']))(x)
f5_2 = (interp1d(pp5_2['nb_steps'], pp5_2['episode_reward']))(x)
f5_3 = (interp1d(pp5_3['nb_steps'], pp5_3['episode_reward']))(x)
exp5 = np.vstack((f5_1, f5_2, f5_3))
f_avg5 = np.average(exp5, axis=0)



pyplot.figure(num=1, figsize=(20, 10),)
pyplot.xlabel('total steps', fontsize=24)
pyplot.ylabel('rewards per episode', fontsize=24)
pyplot.title('"actor_lr" and "target_net update interval" comparison-reward', fontsize=24)

pyplot.plot(f_avg1, 'red', label='actor_lr=0.00001, target_update=0.001')
pyplot.plot(f_avg2, 'b', label='actor_lr=0.001, target_update=0.001')
pyplot.plot(f_avg5, 'g', label='actor_lr=0.00001, target_update=0.01')
pyplot.legend()
pyplot.savefig('save/pics/ddpg_planeball_reward.png',bbox_inches='tight')


pyplot.figure(num=2, figsize=(8, 5),)
width = 0.2
x = np.arange(3)
duration_compare = [duration1,duration2,duration5]
fig, ax = pyplot.subplots()
rects1 = ax.bar(x, duration_compare, width)

pyplot.xticks(x, ('0.00001, 0.001', '0.001, 0.001', '0.00001, 0.01'))
ax.set_ylabel('Duration')
ax.set_title('Training time')
pyplot.savefig('save/pics/ddpg_planeball_duration.png',bbox_inches='tight')

pyplot.show()
