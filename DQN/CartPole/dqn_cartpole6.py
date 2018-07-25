import numpy as np
import gym
import json
import threading

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

#from rl.agents.dqn import DQNAgent
from DQN.dqn import DQNAgent
from DQN.policy import BoltzmannQPolicy, EpsGreedyQPolicy, EpsDisGreedyQPolicy
from DQN.memory import SequentialMemory
import threading
from matplotlib import pyplot
from keras.models import model_from_json
from DQN.callbacks import TestLogger, TrainEpisodeLogger, TrainIntervalLogger, Visualizer, CallbackList, FileLogger

from datetime import datetime
timenow = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

ENV_NAME = 'CartPole-v0'
env = gym.make(ENV_NAME)
np.random.seed(99)
env.seed(99)
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

callback8 = FileLogger(filepath='save/history8_{}'.format(timenow), interval=1)


dqn8 = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, gamma=0.99, nb_steps_warmup=2000,
                target_model_update=1e-2, policy=policy2)
dqn8.compile(Adam(lr=1e-3), metrics=['mae'])
history8 = dqn8.fit(env, nb_epsteps=3000, visualize=False, callbacks=[callback8], verbose=2)

