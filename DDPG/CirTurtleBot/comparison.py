import numpy as np
import gym
import json


from scipy.interpolate import interp1d
from matplotlib import pyplot
from matplotlib.ticker import FuncFormatter
from keras.models import model_from_json


### actor_lr comparison: 0.00001, 0.001, critic_lr comparison: 0.001, 0.01, target_net comparison: 0.001, 0.01

#history1: actor_lr=0.0001, critic_lr=0.001, batch_size=32, memory=50000, target_update=0.001, discount_rate=0.99, warm_up=1000
with open('save/history1_2018-07-31 08:22:20', 'r') as f:
    pp1_1 = json.load(f)
    f.close()
with open('save/history1_2018-07-31 09:42:01', 'r') as f:
    pp1_2 = json.load(f)
    f.close()
with open('save/history1_2018-07-20 10:43:07', 'r') as f:
    pp1_3 = json.load(f)
    f.close()
#with open('save/history1_2018-07-12 18:59:32', 'r') as f:
#    pp1_4 = json.load(f)
#    f.close()
#with open('save/history1_2018-07-12 18:59:37', 'r') as f:
#    pp1_5 = json.load(f)
#    f.close()


#history2: actor_lr=0.00001, critic_lr=0.001, batch_size=32, memory=50000, target_update=0.001, discount_rate=0.99, warm_up=1000

with open('save/history2_2018-07-30 23:25:44', 'r') as f:
    pp2_1 = json.load(f)
    f.close()

with open('save/history2_2018-07-31 01:02:28', 'r') as f:
    pp2_2 = json.load(f)
    f.close()

with open('save/history2_2018-07-31 03:02:58', 'r') as f:
    pp2_3 = json.load(f)
    f.close()


#history3: actor_lr=0.00001, critic_lr=0.01, batch_size=32, memory=50000, target_update=0.001, discount_rate=0.99, warm_up=1000
with open('save/history3_2018-07-30 17:58:48', 'r') as f:
    pp3_1 = json.load(f)
    f.close()

with open('save/history3_2018-07-30 19:15:09', 'r') as f:
    pp3_2 = json.load(f)
    f.close()

#with open('save/history3_2018-07-18 00:29:49', 'r') as f:
#    pp3_3 = json.load(f)
#    f.close()

#with open('save/history3_2018-07-17 15:28:27', 'r') as f:
#    pp3_4 = json.load(f)
#    f.close()

#with open('save/history3_2018-07-17 15:28:35', 'r') as f:
#    pp3_5 = json.load(f)
#    f.close()




duration1 = (sum(pp1_1['duration'])+sum(pp1_2['duration']))/2
duration2 = (sum(pp2_1['duration'])+sum(pp2_2['duration'])+sum(pp2_3['duration']))/3
duration3 = (sum(pp3_1['duration'])+sum(pp3_2['duration']))/2

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




  ### kind: 'linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic, ‘cubic’
x = np.linspace(600, 199000, num=200000)

f1_1 = (interp1d(pp1_1['nb_steps'], pp1_1['episode_reward']))(x)
f1_2 = (interp1d(pp1_2['nb_steps'], pp1_2['episode_reward']))(x)
f1_3 = (interp1d(pp1_3['nb_steps'], pp1_3['episode_reward']))(x)
exp1 = np.vstack((f1_1, f1_2))
f_avg1 = np.average(exp1, axis=0)



f2_1 = (interp1d(pp2_1['nb_steps'], pp2_1['episode_reward']))(x)
f2_2 = (interp1d(pp2_2['nb_steps'], pp2_2['episode_reward']))(x)
f2_3 = (interp1d(pp2_3['nb_steps'], pp2_3['episode_reward']))(x)
exp2 = np.vstack((f2_1, f2_2, f2_3))
f_avg2 = np.average(exp2, axis=0)



f3_1 = (interp1d(pp3_1['nb_steps'], pp3_1['episode_reward']))(x)
f3_2 = (interp1d(pp3_2['nb_steps'], pp3_2['episode_reward']))(x)
#f3_3 = (interp1d(pp3_3['nb_steps'], pp3_3['episode_reward']))(x)
exp3 = np.vstack((f3_1, f3_2))
f_avg3 = np.average(exp3, axis=0)





pyplot.figure(num=1, figsize=(20, 10),)
pyplot.xlabel('total steps', fontsize=24)
pyplot.ylabel('rewards per episode', fontsize=24)
pyplot.title('"actor_lr" and "target_net update interval" comparison-reward', fontsize=24)

pyplot.plot(f_avg1, 'red', label='actor_lr=0.0001, critic_lr=0.001')
pyplot.plot(f_avg2, 'b', label='actor_lr=0.00001, critic_lr=0.001')
pyplot.plot(f_avg3, 'g', label='actor_lr=0.00001, critic_lr=0.01')
pyplot.legend()
pyplot.savefig('save/pics/ddpg_cirturtlebot_reward.png',bbox_inches='tight')


pyplot.figure(num=2, figsize=(8, 5),)
width = 0.2
x = np.arange(3)
duration_compare = [duration1,duration2,duration3]
fig, ax = pyplot.subplots()
rects1 = ax.bar(x, duration_compare, width)

pyplot.xticks(x, ('actor_lr=0.0001,\ncritic_lr=0.001', 'actor_lr=0.0001,\ncritic_lr=0.001', 'actor_lr=0.00001,\ncritic_lr=0.01'))
ax.set_ylabel('Duration')
ax.set_title('Training time')
pyplot.savefig('save/pics/ddpg_cirturtlebot_duration.png',bbox_inches='tight')

pyplot.show()
