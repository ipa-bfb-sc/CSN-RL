import numpy as np
import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam,SGD

from DDPG.ddpg import DDPGAgent
from common.memory import SequentialMemory
from common.random import OrnsteinUhlenbeckProcess
from common.callbacks import TestLogger, TrainEpisodeLogger, TrainIntervalLogger, Visualizer, CallbackList, FileLogger
import environments


from datetime import datetime
timenow = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


#ENV_NAME = 'pendulum-v1'
ENV_NAME = 'ContinuousCartPole-v0'
#ENV_NAME = 'PlaneBall-v0'


gym.undo_logger_setup()

env = gym.make(ENV_NAME)
np.random.seed(88)
env.seed(88)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

# Next, we build a very simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(24))
actor.add(Activation('relu'))
actor.add(Dense(24))
actor.add(Activation('relu'))
actor.add(Dense(24))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('linear'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(24)(x)
x = Activation('relu')(x)
x = Dense(24)(x)
x = Activation('relu')(x)
x = Dense(24)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

model_actor = actor.to_json()
with open("save/Actor.json", "w") as json_file:
    json_file.write(model_actor)
model_critic = critic.to_json()
with open("save/Critic.json", "w") as json_file:
    json_file.write(model_critic)
print("Saved model to disk!")


callback1 = FileLogger(filepath='save/history1_{}'.format(timenow), interval=1)
#callback2 = FileLogger(filepath='save/history2_{}'.format(timenow), interval=1)
#callback3 = FileLogger(filepath='save/history3_{}'.format(timenow), interval=1)
#callback4 = FileLogger(filepath='save/history4_{}'.format(timenow), interval=1)


memory = SequentialMemory(limit=50000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000, batch_size=96,
                  random_process=random_process, gamma=.99, target_model_update=1e-3)
agent.compile([Adam(lr=0.00001),Adam(lr=0.001)], metrics=['mae'])

agent.fit(env, nb_epsteps=3000, visualize=False, callbacks=[callback1], verbose=2)

agent.save_weights('save/ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
print('weights saved!')

#agent.test(env, nb_episodes=500, visualize=True)
