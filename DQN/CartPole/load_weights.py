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
from keras.models import model_from_json

# load json and create model
json_file = open('save/NNmodel1.json', 'r')
loaded_model1 = json_file.read()
json_file.close()
model = model_from_json(loaded_model1)
print("Loaded model from disk!")

ENV_NAME = 'CartPole-v0'
env = gym.make(ENV_NAME)
nb_actions = env.action_space.n
memory = SequentialMemory(limit=5000, window_length=1)
policy = BoltzmannQPolicy()
dqn1 = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000,
               target_model_update=1e-2, policy=policy)
dqn2 = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000,
               target_model_update=1e-2)
dqn3 = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000,
               target_model_update=1e-2, policy=policy, enable_double_dqn=False)
dqn4 = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000,
               target_model_update=1e-2, enable_double_dqn=False)

dqn1.compile(SGD, metrics=['mae'])
dqn2.compile(SGD, metrics=['mae'])
dqn3.compile(SGD, metrics=['mae'])
dqn4.compile(SGD, metrics=['mae'])


dqn1.load_weights('save/dqn1_{}_weights.h5f'.format(ENV_NAME))
dqn2.load_weights('save/dqn2_{}_weights.h5f'.format(ENV_NAME))
dqn3.load_weights('save/dqn3_{}_weights.h5f'.format(ENV_NAME))
dqn4.load_weights('save/dqn4_{}_weights.h5f'.format(ENV_NAME))
print('Weights loaded!')


#test1 = dqn1.test(env, nb_episodes=50, visualize=True)
#test2 = dqn2.test(env, nb_episodes=50, visualize=True)
#test3 = dqn3.test(env, nb_episodes=50, visualize=True)
#test4 = dqn4.test(env, nb_episodes=50, visualize=True)

#pyplot.subplot(2, 1, 1)
#pyplot.plot(test1.history['episode_reward'], 'r--', test2.history['episode_reward'], 'g', test3.history['episode_reward'], 'b--', test4.history['episode_reward'], 'y')

with open('save/history1_2018-06-08 14:53:59', 'r') as f:
    pp1_1 = json.load(f)
    f.close()
with open('save/history1_2018-06-08 15:17:36', 'r') as f:
    pp1_2 = json.load(f)
    f.close()  
with open('save/history1_2018-06-08 15:57:19', 'r') as f:
    pp1_3 = json.load(f)
    f.close()    
with open('save/history1_2018-06-08 16:01:05', 'r') as f:
    pp1_4 = json.load(f)
    f.close()
with open('save/history1_2018-06-08 16:06:10', 'r') as f:
    pp1_5 = json.load(f)
    f.close()

with open('save/history2_2018-06-08 15:17:36', 'r') as f:
    pp2_1 = json.load(f)
    f.close()
with open('save/history2_2018-06-08 14:57:37', 'r') as f:
    pp2_2 = json.load(f)
    f.close()
with open('save/history2_2018-06-08 15:57:19', 'r') as f:
    pp2_3 = json.load(f)
    f.close()
with open('save/history2_2018-06-08 16:01:05', 'r') as f:
    pp2_4 = json.load(f)
    f.close()
with open('save/history2_2018-06-08 16:06:10', 'r') as f:
    pp2_5 = json.load(f)
    f.close()

with open('save/history3_2018-06-08 14:53:59', 'r') as f:
    pp3_1 = json.load(f)
    f.close()
with open('save/history3_2018-06-08 15:17:36', 'r') as f:
    pp3_2 = json.load(f)
    f.close()
with open('save/history3_2018-06-08 15:57:19', 'r') as f:
    pp3_3 = json.load(f)
    f.close()
with open('save/history3_2018-06-08 16:01:05', 'r') as f:
    pp3_4 = json.load(f)
    f.close()
with open('save/history3_2018-06-08 16:06:10', 'r') as f:
    pp3_5 = json.load(f)
    f.close()

with open('save/history4_2018-06-08 14:53:59', 'r') as f:
    pp4_1 = json.load(f)
    f.close()
with open('save/history4_2018-06-08 15:17:36', 'r') as f:
    pp4_2 = json.load(f)
    f.close()
with open('save/history4_2018-06-08 15:57:19', 'r') as f:
    pp4_3 = json.load(f)
    f.close()
with open('save/history4_2018-06-08 16:01:05', 'r') as f:
    pp4_4 = json.load(f)
    f.close()
with open('save/history4_2018-06-08 16:06:10', 'r') as f:
    pp4_5 = json.load(f)
    f.close()


#print(len(pp1_1['nb_steps']), len(pp1_2['nb_steps']),len(pp1_3['nb_steps']),len(pp1_4['nb_steps']),len(pp1_5['nb_steps']),)

er_ave1 = [(pp1_1['episode_reward'][i] + pp1_2['episode_reward'][i]+pp1_3['episode_reward'][i]+pp1_4['episode_reward'][i]+pp1_5['episode_reward'][i])/5 for i in range(len(pp1_1['episode_reward']))]
er_ave2 = [(pp2_1['episode_reward'][i] + pp2_2['episode_reward'][i]+pp2_3['episode_reward'][i]+pp2_4['episode_reward'][i]+pp2_5['episode_reward'][i])/5 for i in range(len(pp2_1['episode_reward']))]
er_ave3 = [(pp3_1['episode_reward'][i] + pp3_2['episode_reward'][i]+pp3_3['episode_reward'][i]+pp3_4['episode_reward'][i]+pp3_5['episode_reward'][i])/5 for i in range(len(pp3_1['episode_reward']))]
er_ave4 = [(pp4_1['episode_reward'][i] + pp4_2['episode_reward'][i]+pp4_3['episode_reward'][i]+pp4_4['episode_reward'][i]+pp4_5['episode_reward'][i])/5 for i in range(len(pp4_1['episode_reward']))]

#tt1 = np.average(np.sum(pp1_1['duration'], pp1_2['duration'], pp1_3['duration'], pp1_4['duration'], pp1_5['duration']))
#tt1 = np.sum(pp1_1['duration'], pp1_2['duration'], pp1_3['duration'], pp1_4['duration'], pp1_5['duration'], axis=0)


zipped_list = zip(pp1_1['duration'], pp1_2['duration'], pp1_3['duration'], pp1_4['duration'], pp1_5['duration'])
t1 = []
t2 = []
t3 = []
t4 = []

t1.append(sum(pp1_1['duration']))
t1.append(sum(pp1_2['duration']))
t1.append(sum(pp1_3['duration']))
t1.append(sum(pp1_4['duration']))
t1.append(sum(pp1_5['duration']))
tt1 = np.average(t1)

t2.append(sum(pp2_1['duration']))
t2.append(sum(pp2_2['duration']))
t2.append(sum(pp2_3['duration']))
t2.append(sum(pp2_4['duration']))
t2.append(sum(pp2_5['duration']))
tt2 = np.average(t2)

t3.append(sum(pp3_1['duration']))
t3.append(sum(pp3_2['duration']))
t3.append(sum(pp3_3['duration']))
t3.append(sum(pp3_4['duration']))
t3.append(sum(pp3_5['duration']))
tt3 = np.average(t3)

t4.append(sum(pp1_1['duration']))
t4.append(sum(pp4_2['duration']))
t4.append(sum(pp4_3['duration']))
t4.append(sum(pp4_4['duration']))
t4.append(sum(pp4_5['duration']))
tt4 = np.average(t4)




print('average "episode_reward" of 5 learning process in 100 episodes:')
print('Experiment1: Policy = Boltzmann, EnableDDQN, Duration:{}'.format(tt1))
print('Experiment2: Policy = EpsGreedy, EnableDDQN, Duration:{}'.format(tt2))
print('Experiment3: Policy = Boltzmann, DisableDDQN, Duration:{}'.format(tt3))
print('Experiment4: Policy = EpsGreedy, DisableDDQN, Duration:{}'.format(tt4))
# x_label is "nb_steps", y_label is "episode_rewards"

pyplot.xlabel('episodes')
pyplot.ylabel('rewards per episode')
pyplot.plot(er_ave1, 'r', label='Boltzmann, EnableDDQN')
pyplot.plot(er_ave2, 'g', label='EpsGreedy, EnableDDQN')
pyplot.plot(er_ave3, 'b', label='Boltzmann, DisableDDQN')
pyplot.plot(er_ave4, 'y', label='EpsGreedy, DisableDDQN')
pyplot.legend()
pyplot.show()




