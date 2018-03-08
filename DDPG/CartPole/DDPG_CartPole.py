"""
Implementation of DDPG - Deep Deterministic Policy Gradient

Algorithm and hyperparameter details can be found here:
    http://arxiv.org/pdf/1509.02971v2.pdf

The algorithm is tested on the Pendulum-v0 OpenAI gym task
and developed with tflearn + Tensorflow

Author: Patrick Emami
"""
import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import tflearn

from replay_buffer import ReplayBuffer
import os


# use correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # use correct GPU


# ==========================
#   Training Parameters
# ==========================
# Max training steps
MAX_EPISODES = 1000
# Max episode length
MAX_EP_STEPS = 200
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.01
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Discount factor
GAMMA = 0.99
# Soft target update param
TAU = 0.05
STATE_DIM = 4
ACTION_DIM = 1
ACTION_PROB_DIMS = 2
ACTION_BOUND = 1
ACTION_SPACE = [0, 1]

# ===========================
#   Utility Parameters
# ===========================
# Render gym env during training
RENDER_ENV = True
# Use Gym Monitor
GYM_MONITOR_EN = True
# Gym environment
ENV_NAME = 'CartPole-v0'
# Directory for storing gym results
MONITOR_DIR = './results/gym_ddpg'
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/tf_ddpg'
RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 10000
MINIBATCH_SIZE = 64


# ===========================
#   Actor and Critic DNNs
# ===========================
class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -2 and 2
    """

    def __init__(self, sess):
        self.sess = sess
        self.s_dim = STATE_DIM
        self.a_dim = ACTION_DIM
        self.a_prob_dim = ACTION_PROB_DIMS
        self.action_bound = ACTION_BOUND
        self.learning_rate = ACTOR_LEARNING_RATE
        self.tau = TAU

        # Actor Network
        self.onnet_in_states, self.out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + \
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.actor_gradients = tf.gradients(self.out, self.network_params, -self.action_gradient)

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate). \
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        in_states = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(in_states, 400, activation='relu')
        net = tflearn.fully_connected(net, 300, activation='relu')
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out_actions = tflearn.fully_connected(net, ACTION_PROB_DIMS, activation='softmax', weights_init=w_init)
        return in_states, out_actions

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.onnet_in_states: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inp_states):
        out_actions = self.sess.run(self.out, feed_dict={
            self.onnet_in_states: inp_states
        })
        out_actions = out_actions[0]
        #print("actor output actions", out_actions)
        return out_actions

    def predict_target(self, in_states):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: in_states
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, sess, num_actor_vars):
        self.sess = sess
        self.s_dim = STATE_DIM
        self.a_dim = ACTION_DIM
        self.learning_rate = CRITIC_LEARNING_RATE
        self.tau = TAU

        # Create the critic network
        self.in_states, self.in_actions, self.onnet_out_reward = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(
                tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_values = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_values, self.onnet_out_reward)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action
        self.action_grads = tf.gradients(self.onnet_out_reward, self.in_actions)

    def create_critic_network(self):
        inp_state = tflearn.input_data(shape=[None, self.s_dim])
        inp_action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inp_state, 400, activation='relu')

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 300)
        t2 = tflearn.fully_connected(inp_action, 300)

        net = tflearn.activation(tf.matmul(net, t1.W) + tf.matmul(inp_action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        #out_rewards = tflearn.layers.core.single_unit(net, activation='linear', name='output_rewards')
        out_reward = tflearn.fully_connected(net, 1, weights_init=w_init)  # FIXME predicts single reward, need string of rewards

        return inp_state, inp_action, out_reward

    def train(self, observed_states, observed_action, mixed_rewards):  # note: replaced predicted_q_value with sum of mixed rewards
        return self.sess.run([self.onnet_out_reward, self.optimize], feed_dict={
            self.in_states: observed_states,
            self.in_actions: observed_action,
            self.predicted_q_values: mixed_rewards
        })

    def predict(self, inputs, action):
        return self.sess.run(self.onnet_out_reward, feed_dict={
            self.in_states: inputs,
            self.in_actions: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.in_states: inputs,
            self.in_actions: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)


# ===========================
#   Tensorflow Summary Ops
# ===========================
def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars


# ===========================
#   Agent Training
# ===========================
def train(sess, env, actor, critic):
    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

    for i in range(MAX_EPISODES):

        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0

        for j in range(MAX_EP_STEPS):

            if RENDER_ENV:
                env.render()

            action_probabilities = actor.predict(np.reshape(s, (1, STATE_DIM)))
            #print("action probs", action_probabilities)
            action = choose_action(action_probabilities)
            #print("action", action)
            s2, r, done, info = env.step(action)

            replay_buffer.add(np.reshape(s, (actor.s_dim,)), action, r, \
                              done, np.reshape(s2, (actor.s_dim,)))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > MINIBATCH_SIZE:
                s_batch, a_batch, r_batch, done_batch, s2_batch = \
                    replay_buffer.sample_batch(MINIBATCH_SIZE)

                # action probs to actions  # TODO how to deal with non-determinate policies
                # convert actor.predict_target(s2_batch) to actions
                # the problem is that critic expects actions to always be determinate, when in fact they are probab
                # Calculate targets
                # todo can we just feed real a and s batch here, no s2?
                # fixme critic predict expects 1D actions not 2D probabilities
                a_batch = np.reshape(a_batch, (len(a_batch), 1))
                #print("sbshape", np.shape(s_batch), "\n a shape", np.shape(a_batch))
                targnet_predicted_reward = critic.predict_target(s_batch, a_batch)
                #targnet_predicted_reward = critic.predict_target(s2_batch, actor.predict_target(s2_batch))
                # print("targnet prediction", targnet_predicted_reward)  # this is a whole reward tensor!!

                # actually, we mix observations with predictions by factor gamma
                # fixme I think we need to get rid of this block. targ reward is single value?
                obs_plus_predicted_rewards = []
                for k in range(MINIBATCH_SIZE):
                    if done_batch[k]:
                        obs_plus_predicted_rewards.append(r_batch[k])  # final timestep is just the reward
                    else:
                        obs_plus_predicted_rewards.append(r_batch[k] + GAMMA * targnet_predicted_reward[k])
                obs_plus_predicted_rewards = np.reshape(obs_plus_predicted_rewards, (len(obs_plus_predicted_rewards), 1))
                # Update the critic given the targets
                predicted_q_value, _ = critic.train(s_batch, a_batch, obs_plus_predicted_rewards)
                #predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(observed_rewards, (MINIBATCH_SIZE, 1)))

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                #a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_batch)
                #grads = critic.action_gradients(s_batch, a_outs)  # we aren't deterministic
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            s = s2
            ep_reward += r

            if done:
                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / float(j)
                })

                writer.add_summary(summary_str, i)
                writer.flush()
                # TODO checkwhich ep reward is being printed
                print(  # TODO replace maxq with something more interesting
                '| Reward: %.2i' % int(ep_reward), " | Episode", i, \
                '| Qmax: %.4f' % (ep_ave_max_q / float(j)))

                break

def choose_action(probabilities):
    choice = int(np.random.choice(ACTION_SPACE, 1, p=probabilities))
    return choice

def main(_):
    with tf.Session() as sess:
        # TODO: reduce network sizes. keep all states stop editing this ver, add dropout in successor
        env = gym.make(ENV_NAME)
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)
        env.seed(RANDOM_SEED)

        # Ensure action bound is symmetric
       # assert (env.action_space.high == -env.action_space.low)

        actor = ActorNetwork(sess)

        critic = CriticNetwork(sess, actor.get_num_trainable_vars())

        env = gym.wrappers.Monitor(env, MONITOR_DIR, force=True)

        train(sess, env, actor, critic)



if __name__ == '__main__':
    tf.app.run()