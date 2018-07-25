import numpy as np
import gym
import json

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

#from rl.agents.dqn import DQNAgent
from DQN.dqn import DQNAgent
from DQN.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from DQN.memory import SequentialMemory

from matplotlib import pyplot
from keras.models import model_from_json
from DQN.callbacks import TestLogger, TrainEpisodeLogger, TrainIntervalLogger, Visualizer, CallbackList, FileLogger
import environments

from datetime import datetime
timenow = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

ENV_NAME = 'DiscretePlaneBall-v0'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
#env = dplaneball.DiscretePlaneBallEnv()
np.random.seed(88)
env.seed(88)
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
with open("save/NNmodel2.json", "w") as json_file:
    json_file.write(model_save)

print("Saved model to disk!")


memory = SequentialMemory(limit=100000, window_length=1)
policy1 = BoltzmannQPolicy()
policy2 = EpsGreedyQPolicy()

callback8 = FileLogger(filepath='save/history8_{}'.format(timenow), interval=1)


dqn3 = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, batch_size=640, nb_steps_warmup=20000,
               target_model_update=1e-2, policy=policy2)
dqn3.compile(Adam(lr=1e-2), metrics=['mae'])
history3 = dqn3.fit(env, nb_epsteps=2500, visualize=False, callbacks=[callback8], verbose=2)


#dqn4.save_weights('save/dqn4_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
