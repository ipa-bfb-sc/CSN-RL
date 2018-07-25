import numpy as np
import gym
import json


from scipy.interpolate import interp1d
from matplotlib import pyplot
from matplotlib.ticker import FuncFormatter
from keras.models import model_from_json

## actor_lr: 0.0001, 0.00001; epsilon: 0.2, 0.5

# history 1: actor_lr=0.00001, critic_lr=0.0001, eps=0.2, batch_size=100, a_update_steps=10, c_update_steps=100
with open('save/history1_2018-07-25 09:30:59', 'r') as f:
    pp1_1 = json.load(f)
    f.close()
with open('save/history1_2018-07-25 09:41:51', 'r') as f:
    pp1_2 = json.load(f)
    f.close()
with open('save/history1_2018-07-25 09:48:23', 'r') as f:
    pp1_3 = json.load(f)
    f.close()



# history 2: actor_lr=0.00001, critic_lr=0.0001, eps=0.2, batch_size=200, a_update_steps=10, c_update_steps=100
with open('save/history2_2018-07-25 10:03:33', 'r') as f:
    pp2_1 = json.load(f)
    f.close()

with open('save/history2_2018-07-25 10:09:08', 'r') as f:
    pp2_2 = json.load(f)
    f.close()

with open('save/history2_2018-07-25 10:14:34', 'r') as f:
    pp2_3 = json.load(f)
    f.close()




# history 3: actor_lr=0.00001, critic_lr=0.0001, eps=0.2, batch_size=200, a_update_steps=100, c_update_steps=100
with open('save/history3_2018-07-25 12:09:14', 'r') as f:
    pp3_1 = json.load(f)
    f.close()

with open('save/history3_2018-07-25 12:16:50', 'r') as f:
    pp3_2 = json.load(f)
    f.close()

with open('save/history3_2018-07-25 12:23:53', 'r') as f:
    pp3_3 = json.load(f)
    f.close()

'''


# history 4: actor_lr=0.00001, critic_lr=0.00002, eps=0.2, batch_size=320, update_steps=10
with open('save/history4_2018-07-23 14:51:50', 'r') as f:
    pp4_1 = json.load(f)
    f.close()

with open('save/history4_2018-07-23 14:55:55', 'r') as f:
    pp4_2 = json.load(f)
    f.close()

with open('save/history4_2018-07-23 15:01:16', 'r') as f:
    pp4_3 = json.load(f)
    f.close()



# history 5: actor_lr=0.0001, critic_lr=0.00002, eps=0.2, batch_size=32, update_steps=10
with open('save/history5_2018-07-23 15:57:27', 'r') as f:
    pp5_1 = json.load(f)
    f.close()

with open('save/history5_2018-07-23 16:09:26', 'r') as f:
    pp5_2 = json.load(f)
    f.close()

with open('save/history5_2018-07-23 16:17:35', 'r') as f:
    pp5_3 = json.load(f)
    f.close()

'''
duration1 = (pp1_1['duration_total']+pp1_2['duration_total']+pp1_3['duration_total'])/3
duration2 = (pp2_1['duration_total']+pp2_2['duration_total']+pp2_3['duration_total'])/3
duration3 = (pp3_1['duration_total']+pp3_2['duration_total']+pp3_3['duration_total'])/3
#duration4 = (pp4_1['duration_total']+pp4_2['duration_total']+pp4_3['duration_total'])/3
#duration5 = pp5_1['duration_total']




#print('duration1:{}, duration5:{}, duration3:{}, duration4:{}'.format(duration1, duration5, duration3, duration4))



  ### kind: 'linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic, ‘cubic’
x = np.linspace(2000, 498000, num=500000)

f1_1 = (interp1d(pp1_1['steps'], pp1_1['reward']))(x)
f1_2 = (interp1d(pp1_2['steps'], pp1_2['reward']))(x)
f1_3 = (interp1d(pp1_3['steps'], pp1_3['reward']))(x)

exp1 = np.vstack((f1_1, f1_2, f1_3))
f_avg1 = np.average(exp1, axis=0)



f2_1 = (interp1d(pp2_1['steps'], pp2_1['reward']))(x)
f2_2 = (interp1d(pp2_2['steps'], pp2_2['reward']))(x)
f2_3 = (interp1d(pp2_3['steps'], pp2_3['reward']))(x)
#f2_4 = (interp1d(pp2_4['steps'], pp2_4['reward']))(x)
#f2_5 = (interp1d(pp2_5['steps'], pp2_5['reward']))(x)
exp2 = np.vstack((f2_1, f2_2, f2_3))
f_avg2 = np.average(exp2, axis=0)



f3_1 = (interp1d(pp3_1['steps'], pp3_1['reward']))(x)
f3_2 = (interp1d(pp3_2['steps'], pp3_2['reward']))(x)
f3_3 = (interp1d(pp3_3['steps'], pp3_3['reward']))(x)
#f3_4 = (interp1d(pp3_4['steps'], pp3_4['reward']))(x)
#f3_5 = (interp1d(pp3_5['steps'], pp3_5['reward']))(x)
exp3 = np.vstack((f3_1, f3_2, f3_3))
f_avg3 = np.average(exp3, axis=0)

'''
f4_1 = (interp1d(pp4_1['steps'], pp4_1['reward']))(x)
f4_2 = (interp1d(pp4_2['steps'], pp4_2['reward']))(x)
f4_3 = (interp1d(pp4_3['steps'], pp4_3['reward']))(x)
exp4 = np.vstack((f4_1, f4_2, f4_3))
f_avg4 = np.average(exp4, axis=0)


f5_1 = (interp1d(pp5_1['steps'], pp5_1['reward']))(x)
f5_2 = (interp1d(pp5_2['steps'], pp5_2['reward']))(x)
f5_3 = (interp1d(pp5_3['steps'], pp5_3['reward']))(x)
exp5 = np.vstack((f5_1, f5_2, f5_3))
f_avg5 = np.average(exp5, axis=0)
'''

pyplot.figure(num=1, figsize=(20, 10),)
pyplot.xlabel('total steps', fontsize=24)
pyplot.ylabel('rewards per episode', fontsize=24)
pyplot.title('"loop update steps" and "batch size" comparison-reward', fontsize=24)

pyplot.plot(f_avg1, 'red', label='batch_size=100, update_steps=(10,100)')
pyplot.plot(f_avg2, 'b', label='batch_size=200, update_steps=(10,100)')
pyplot.plot(f_avg3, 'g', label='batch_size=200, update_steps=(100,100)', alpha=0.7)
pyplot.legend()
pyplot.savefig('save/pics/ppo_planeball_reward.png',bbox_inches='tight')



pyplot.figure(num=2, figsize=(8, 5),)
width = 0.2
x = np.arange(3)
duration_compare = [duration1,duration2,duration3]
fig, ax = pyplot.subplots()
rects1 = ax.bar(x, duration_compare, width)

pyplot.xticks(x, ('batch_size=100,\nupdate_steps=(10,100)', 'batch_size=200,\nupdate_steps=(10,100)', 'batch_size=100,\nupdate_steps=(100,100)'))
ax.set_ylabel('Duration')
ax.set_title('Training time')
pyplot.savefig('save/pics/ppo_planeball_duration.png',bbox_inches='tight')


pyplot.show()
