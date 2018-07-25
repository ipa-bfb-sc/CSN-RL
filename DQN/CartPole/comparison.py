import numpy as np
import gym
import json


from scipy.interpolate import interp1d
from matplotlib import pyplot
from matplotlib.ticker import FuncFormatter
from keras.models import model_from_json


### policy: boltzmann, diseps, batch_size:32, 64, 96, discount_rate: 0.1, 0.5, 0.99

# history 1: Boltzmann, batch_size=64, memory=10000, discount_rate=0.99
with open('save/nhistory1_2018-07-18 12:39:53', 'r') as f:
    pp1_1 = json.load(f)
    f.close()
with open('save/nhistory1_2018-07-18 12:40:06', 'r') as f:
    pp1_2 = json.load(f)
    f.close()
with open('save/nhistory1_2018-07-18 12:40:20', 'r') as f:
    pp1_3 = json.load(f)
    f.close()
#with open('save/history1_2018-07-12 18:59:32', 'r') as f:
#    pp1_4 = json.load(f)
#    f.close()
#with open('save/history1_2018-07-12 18:59:37', 'r') as f:
#    pp1_5 = json.load(f)
#    f.close()


# history 2: Boltzmann, batch_size=64, memory=10000, discount_rate=0.5
with open('save/nhistory2_2018-07-18 12:52:42', 'r') as f:
    pp2_1 = json.load(f)
    f.close()

with open('save/nhistory2_2018-07-18 12:53:11', 'r') as f:
    pp2_2 = json.load(f)
    f.close()

with open('save/nhistory2_2018-07-18 12:53:30', 'r') as f:
    pp2_3 = json.load(f)
    f.close()


# history 3: DisEpsGreedy, batch_size=64, memory=10000, discount_rate=0.99
with open('save/nhistory3_2018-07-18 13:27:00', 'r') as f:
    pp3_1 = json.load(f)
    f.close()

with open('save/nhistory3_2018-07-18 13:27:17', 'r') as f:
    pp3_2 = json.load(f)
    f.close()

with open('save/nhistory3_2018-07-18 13:27:35', 'r') as f:
    pp3_3 = json.load(f)
    f.close()

#with open('save/history3_2018-07-17 15:28:27', 'r') as f:
#    pp3_4 = json.load(f)
#    f.close()

#with open('save/history3_2018-07-17 15:28:35', 'r') as f:
#    pp3_5 = json.load(f)
#    f.close()

'''
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
'''

duration1 = (sum(pp1_1['duration'])+sum(pp1_2['duration'])+sum(pp1_3['duration']))/3
duration2 = (sum(pp2_1['duration'])+sum(pp2_2['duration'])+sum(pp2_3['duration']))/3
duration3 = (sum(pp3_1['duration'])+sum(pp3_2['duration'])+sum(pp3_3['duration']))/3
#duration4 = (sum(pp4_1['duration'])+sum(pp4_2['duration'])+sum(pp4_3['duration']))/3
#duration5 = (sum(pp5_1['duration'])+sum(pp5_2['duration'])+sum(pp5_3['duration']))/3




reward1 = (sum(pp1_1['episode_reward'])+sum(pp1_2['episode_reward'])+sum(pp1_3['episode_reward']))/3
reward2 = (sum(pp3_1['episode_reward'])+sum(pp3_2['episode_reward'])+sum(pp3_3['episode_reward']))/3
print(reward1, reward2)



  ### kind: 'linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic, ‘cubic’
x = np.linspace(200, 299800, num=300000)

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

'''
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

'''

pyplot.figure(num=1, figsize=(20, 10),)
pyplot.xlabel('total steps', fontsize=24)
pyplot.ylabel('rewards per episode', fontsize=24)
pyplot.title('"exploration strategy" and "discounted rate" comparison-reward', fontsize=24)

pyplot.plot(f_avg1, 'red', label='Boltzmann, discount_rate=0.99')
pyplot.plot(f_avg2, 'b', label='Boltzmann, discount_rate=0.5')
pyplot.plot(f_avg3, 'g', label='DisEpsGreedy, discount_rate=0.99', alpha=0.7)
pyplot.legend()
pyplot.savefig('save/pics/dqn_cartpole_reward.png',bbox_inches='tight')


pyplot.figure(num=2, figsize=(8, 5),)
width = 0.2
x = np.arange(3)
duration_compare = [duration1,duration2,duration3]
fig, ax = pyplot.subplots()
rects1 = ax.bar(x, duration_compare, width)

pyplot.xticks(x, ('Boltzmann, 0.99', 'Boltzmann, 0.5', 'DisEpsGreedy, 0.99'))
ax.set_ylabel('Duration')
ax.set_title('Training time')
pyplot.savefig('save/pics/dqn_cartpole_duration.png',bbox_inches='tight')


pyplot.show()
