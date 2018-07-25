import numpy as np
import gym
import json
import threading

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

#from rl.agents.dqn import DQNAgent
from DQN.dqn import DQNAgent
from common.policy import BoltzmannQPolicy, EpsGreedyQPolicy, EpsDisGreedyQPolicy
from common.memory import SequentialMemory
import threading
from matplotlib import pyplot
from keras.models import model_from_json
from common.callbacks import TestLogger, TrainEpisodeLogger, TrainIntervalLogger, Visualizer, CallbackList, FileLogger

from datetime import datetime
timenow = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

ENV_NAME = 'CartPole-v0'
env = gym.make(ENV_NAME)
np.random.seed(77)
env.seed(77)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(24))
model.add(Activation('relu'))
model.add(Dense(24))
model.add(Activation('relu'))
model.add(Dense(24))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# serialize model to JSON
model_save = model.to_json()
with open("save/NNmodel1.json", "w") as json_file:
    json_file.write(model_save)

print("Saved model to disk!")



# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=10000, window_length=1)
#policy1 = BoltzmannQPolicy()
policy1 = EpsDisGreedyQPolicy(eps=0.0001, eps_decay=0.999)
policy2 = BoltzmannQPolicy()
callback1 = FileLogger(filepath='save/nhistory1_{}'.format(timenow), interval=1)
callback2 = FileLogger(filepath='save/nhistory2_{}'.format(timenow), interval=1)
callback3 = FileLogger(filepath='save/nhistory3_{}'.format(timenow), interval=1)
callback4 = FileLogger(filepath='save/nhistory4_{}'.format(timenow), interval=1)



#dqn1 = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, batch_size=64, nb_steps_warmup=2000,
#               target_model_update=1e-2, policy=policy2)
#dqn1.compile(Adam(lr=1e-3), metrics=['mae'])
#dqn1.fit(env, nb_steps=300000, visualize=False, callbacks=[callback1], verbose=2)

#dqn2 = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, batch_size=64, gamma=0.5, nb_steps_warmup=2000,
#               target_model_update=1e-2, policy=policy2)
#dqn2.compile(Adam(lr=1e-3), metrics=['mae'])
#dqn2.fit(env, nb_steps=300000, visualize=False, callbacks=[callback2], verbose=2)

dqn3 = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, batch_size=64, nb_steps_warmup=2000,
               target_model_update=1e-2, policy=policy1)
dqn3.compile(Adam(lr=1e-3), metrics=['mae'])
dqn3.fit(env, nb_steps=300000, visualize=False, callbacks=[callback3], verbose=2)
'''
pyplot.figure()
pyplot.subplot(2, 1, 1)
pyplot.plot(history1.history['episode_reward'], 'r--',history3.history['episode_reward'], 'b--')

pyplot.subplot(2, 1, 2)
#pyplot.plot(history1.history['nb_steps'], history1.history['episode_reward'], 'r', history2.history['nb_steps'], history2.history['episode_reward'], 'g')
pyplot.plot(history2.history['episode_reward'], 'r', history4.history['episode_reward'], 'b')
pyplot.show()

#pyplot.savefig('save/BoltzmannQPolicy')
'''




# After training is done, we save the final weights.
#dqn1.save_weights('save/dqn1_{}_weights_test.h5f'.format(ENV_NAME), overwrite=True)
#dqn2.save_weights('save/dqn2_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
#dqn3.save_weights('save/dqn3_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
#dqn4.save_weights('save/dqn4_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
#print('Weights saved!')

