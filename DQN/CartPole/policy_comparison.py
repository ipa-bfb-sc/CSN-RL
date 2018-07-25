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
# load json and create model
json_file = open('save/NNmodel1.json', 'r')
loaded_model1 = json_file.read()
json_file.close()
model = model_from_json(loaded_model1)
print("Loaded model from disk!")

ENV_NAME = 'CartPole-v0'
env = gym.make(ENV_NAME)

np.random.seed(256)
env.seed(256)

nb_actions = env.action_space.n
memory = SequentialMemory(limit=5000, window_length=1)
policy = BoltzmannQPolicy()
dqn1 = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000,
               target_model_update=1e-2, policy=policy)
dqn2 = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000,
               target_model_update=1e-2)


dqn1.compile(SGD, metrics=['mae'])
dqn2.compile(SGD, metrics=['mae'])



dqn1.load_weights('save/dqn1_Eps(0.001,0.999)_CartPole-v0_weights_test.h5f'.format(ENV_NAME))
dqn2.load_weights('save/dqn2_{}_weights.h5f'.format(ENV_NAME))

print('Weights loaded!')


#test1 = dqn1.test(env, nb_episodes=500, visualize=True)
#test2 = dqn2.test(env, nb_episodes=500, visualize=True)
'''

'''
with open('save/history2_2018-06-19 11:57:31', 'r') as f:
    pp1_1 = json.load(f)
    f.close()

with open('save/history2_2018-06-19 12:18:46', 'r') as f:
    pp1_2 = json.load(f)
    f.close()

with open('save/history2_2018-06-19 12:33:24', 'r') as f:
    pp1_3 = json.load(f)
    f.close()

with open('save/history2_2018-06-19 12:46:40', 'r') as f:
    pp1_4 = json.load(f)
    f.close()

with open('save/history2_2018-06-19 13:39:49', 'r') as f:
    pp1_5 = json.load(f)
    f.close()


with open('save/history1_2018-06-19 13:57:26', 'r') as f:
    pp2_1 = json.load(f)
    f.close()

with open('save/history1_2018-06-19 14:12:47', 'r') as f:
    pp2_2 = json.load(f)
    f.close()

with open('save/history1_2018-06-19 14:25:52', 'r') as f:
    pp2_3 = json.load(f)
    f.close()

with open('save/history1_2018-06-19 14:40:00', 'r') as f:
    pp2_4 = json.load(f)
    f.close()

with open('save/history1_2018-06-19 14:54:44', 'r') as f:
    pp2_5 = json.load(f)
    f.close()
'''

with open('save/history9_2018-06-20 10:50:57', 'r') as f:
    pp1_1 = json.load(f)
    f.close()

with open('save/history9_2018-06-20 11:55:39', 'r') as f:
    pp1_2 = json.load(f)
    f.close()

with open('save/history9_2018-06-20 12:56:56', 'r') as f:
    pp1_3 = json.load(f)
    f.close()

with open('save/history9_2018-06-20 13:59:14', 'r') as f:
    pp1_4 = json.load(f)
    f.close()

with open('save/history9_2018-06-20 14:28:07', 'r') as f:
    pp1_5 = json.load(f)
    f.close()


with open('save/history10_2018-06-20 10:51:04', 'r') as f:
    pp2_1 = json.load(f)
    f.close()

with open('save/history10_2018-06-20 11:58:08', 'r') as f:
    pp2_2 = json.load(f)
    f.close()

with open('save/history10_2018-06-20 12:56:58', 'r') as f:
    pp2_3 = json.load(f)
    f.close()

with open('save/history10_2018-06-20 13:59:28', 'r') as f:
    pp2_4 = json.load(f)
    f.close()

with open('save/history10_2018-06-20 15:01:30', 'r') as f:
    pp2_5 = json.load(f)
    f.close()

duration1 = (sum(pp1_1['duration'])+sum(pp1_2['duration'])+sum(pp1_3['duration'])+sum(pp1_4['duration'])+sum(pp1_5['duration']))/5
duration2 = (sum(pp2_1['duration'])+sum(pp2_2['duration'])+sum(pp2_3['duration'])+sum(pp2_4['duration'])+sum(pp2_5['duration']))/5

print('Duration1:{}, Duration2:{}'.format(duration1,duration2))

er_ave1 = [(pp1_1['episode_reward'][i] + pp1_2['episode_reward'][i]+pp1_3['episode_reward'][i]+pp1_4['episode_reward'][i]+pp1_5['episode_reward'][i])/5 for i in range(len(pp1_1['episode_reward']))]
er_ave2 = [(pp2_1['episode_reward'][i] + pp2_2['episode_reward'][i]+pp2_3['episode_reward'][i]+pp2_4['episode_reward'][i]+pp2_5['episode_reward'][i])/5 for i in range(len(pp2_1['episode_reward']))]
#er_ave3 = [(pp3_1['episode_reward'][i] + pp3_2['episode_reward'][i]+pp3_3['episode_reward'][i]+pp3_4['episode_reward'][i]+pp3_5['episode_reward'][i])/5 for i in range(len(pp3_1['episode_reward']))]
#er_ave4 = [(pp4_1['episode_reward'][i] + pp4_2['episode_reward'][i]+pp4_3['episode_reward'][i]+pp4_4['episode_reward'][i]+pp4_5['episode_reward'][i])/5 for i in range(len(pp4_1['episode_reward']))]

#pyplot.subplot(2, 1, 1)

pyplot.figure(num=1, figsize=(20, 10),)
pyplot.xlabel('episodes', fontsize=24)
pyplot.ylabel('rewards per episode', fontsize=24)
pyplot.title('Exploration strategy comparison', fontsize=24)

new_ticks = np.linspace(0, 3000, 30)
pyplot.xticks(new_ticks)
pyplot.plot(er_ave1, 'r', label='EpsDisGreedy, 1-0.0001/0.999')
pyplot.plot(er_ave2, 'g', label='Boltzmann')
pyplot.legend()
pyplot.savefig('save/pics/cartpole_policy2.png',bbox_inches='tight')


pyplot.figure(num=2, figsize=(8, 5),)
width = 0.2
x = np.arange(2)
duration_compare = [duration1,duration2]
fig, ax = pyplot.subplots()
rects1 = ax.bar(x, duration_compare, width)

pyplot.xticks(x, ('EpsDisGreedy, 1-0.0001/0.999', 'Boltzmann'))
ax.set_ylabel('Duration')
ax.set_title('Training time comparison')
pyplot.savefig('save/pics/cartpole_duration_P.png',bbox_inches='tight')
pyplot.show()
