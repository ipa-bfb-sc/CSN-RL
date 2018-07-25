from keras.layers import Dense, Activation
from keras.models import Model, Sequential
from keras.layers.merge import concatenate


def simple_actor(env):
    """Build a simple actor network"""

    x = Dense(24, activation='relu')(env.state)
    x = Dense(24, activation='relu')(x)
    x = Dense(24, activation='relu')(x)
    x = Dense(env.action_space.dim, activation='linear')(x)
    actor = Model(inputs=[env.state], outputs=[x])
    print(actor.summary())
    return actor

'''
    actor = Sequential()
    actor.add(Dense(24)(env.state))
    actor.add(Activation('relu'))
    actor.add(Dense(24))
    actor.add(Activation('relu'))
    actor.add(Dense(24))
    actor.add(Activation('relu'))
    actor.add(Dense(env.action_space.dim, activation="tanh"))
    actor.add(Activation('linear'))
    print(actor.summary())
    return actor
'''

def simple_critic(env):
    """Build a simple critic network"""
    observation = env.state
    action = env.action
    # Concatenate the inputs for the critic
    inputs = concatenate([observation, action])
    x = Dense(24)(inputs)
    x = Activation('relu')(x)
    x = Dense(24)(x)
    x = Activation('relu')(x)
    x = Dense(24)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[observation, action], outputs=[x])
    print(critic.summary())
    return critic

'''
def simple_actor(env):
    """Build a simple actor network"""
    x = Dense(env.action_space.dim, activation="tanh")(env.state)
    return Model(inputs=[env.state], outputs=[x])


def simple_critic(env):
    """Build a simple critic network"""
    observation = env.state
    action = env.action
    # Concatenate the inputs for the critic
    inputs = concatenate([observation, action])
    x = Dense(1)(inputs)
    x = Activation('linear')(x)

    # Final model
    return Model(inputs=[observation, action], outputs=[x])

'''