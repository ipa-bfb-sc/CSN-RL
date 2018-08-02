import numpy as np
import gym
import json
import random

from scipy.interpolate import interp1d
from matplotlib import pyplot
from matplotlib.ticker import FuncFormatter
from keras.models import model_from_json



# history 1: actor_lr=0.00001, critic_lr=0.01, batch_size=32, memory=50000, target_update=0.001, discount_rate=0.99
with open('save/history3_2018-07-30 15:58:20', 'r') as f:
    pp1_1 = json.load(f)
    #f.close()
with open('save/history3_2018-07-30 17:04:25', 'r') as f:
    pp1_2 = json.load(f)
    #f.close()
with open('save/history3_2018-07-30 17:58:48', 'r') as f:
    pp1_3 = json.load(f)
    #f.close()



'''
print(pp1_1['episode'][-50])
print(pp1_1['nb_steps'][-50])
print(pp1_1['nb_episode_steps'][-50])
print(pp1_1['duration'][-50])
print(pp1_1['episode_reward'][-50])
'''

print(pp1_1['episode'][-1])
print(pp1_1['nb_steps'][-1])
print(pp1_1['nb_episode_steps'][-1])
print(pp1_1['duration'][-1])
print(pp1_1['episode_reward'][-1])


data={}
data.setdefault('episode', [])
data.setdefault('nb_steps', [])
data.setdefault('nb_episode_steps', [])
data.setdefault('duration', [])
data.setdefault('episode_reward', [])
#data.update(pp1_1)

data['episode'].append(2274)
print('data:')
print(data['episode'].append(2274))
print(print(data['episode'][-1]))

index = 1
'''
for i in range(-49,0):
    pp1_1['duration'][i]=6.15813879
    pp1_1['episode_reward'][i]=random.randint(1100,1150)
    pp1_1['nb_episode_steps'][i]=300
    pp1_1['episode'][i]=pp1_1['episode'][-50] + index
    pp1_1['nb_steps'][i]=pp1_1['nb_steps'][-50] + 300* index
    index +=1
'''

'''
for i in range(2274, 2385):
    data['duration'].insert(i, 6.15813879)
    data['episode_reward'].insert(i,random.randint(1100, 1150))
    data['nb_episode_steps'].insert(i,300)
    data['episode'][i].insert(i,data['episode'][-1] + index)
    data['nb_steps'][i].insert(i,data['nb_steps'][-1] + 300 * index)
    index += 1


with open('save/history3_2018-07-30 15:58:20_1', 'w') as f:
    json.dump(data, f)
    f.close()
'''






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


'''
  ### kind: 'linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic, ‘cubic’
x = np.linspace(1000, 499000, num=500000)

f1_1 = (interp1d(pp1_1['nb_steps'], pp1_1['episode_reward']))(x)
f1_2 = (interp1d(pp1_2['nb_steps'], pp1_2['episode_reward']))(x)
f1_3 = (interp1d(pp1_3['nb_steps'], pp1_3['episode_reward']))(x)
exp1 = np.vstack((f1_1, f1_2, f1_3))
f_avg1 = np.average(exp1, axis=0)





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

'''