# OpenGym CartPole-v0 with A3C
# -----------------------------------
#
# A3C implementation with CPU optimizer threads.
#
#
# reference from: Jaromir Janisch
# https://jaromiru.com/2017/02/16/lets-make-an-a3c-theory/

import timeit
import numpy as np
import tensorflow as tf
#The Thread class represents an activity that is run in a separate thread of control.
import gym, time, random, threading

from keras.models import *
from keras.layers import *
from keras import backend as K

#-- constants
ENV = 'CartPole-v0'

RUN_TIME = 80
THREADS = 4		#4 agents
OPTIMIZERS = 2	#2 optimizers
THREAD_DELAY = 0.001

GAMMA = 0.99	#discount rate

N_STEP_RETURN = 8	#to look up 8 states further
GAMMA_N = GAMMA ** N_STEP_RETURN	#Performs exponential (power) calculation

EPS_START = 0.4
EPS_STOP  = .15
EPS_STEPS = 75000

MIN_BATCH = 32	#example batch
LEARNING_RATE = 5e-3	#learning rate

LOSS_V = .5			# v loss coefficient
LOSS_ENTROPY = .01 	# entropy coefficient

#---------
#difine calss Brain: NN model, TensorFlow graph, optimize(minize the loss function which defined in TensorFlow), push examples into train queue
class Brain:
    train_queue = [ [], [], [], [], [] ]	# s, a, r, s', s' terminal mask
    lock_queue = threading.Lock()	#Once a thread has acquired it, subsequent attempts to acquire it block, until it is released

    def __init__(self):
        self.session = tf.Session()
        K.set_session(self.session)
        K.manual_variable_initialization(True)	#whether variables should be initialized as they are instantiated (default), or if the user should handle the initialization

        self.model = self._build_model()
        self.graph = self._build_graph(self.model)

        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()

        self.default_graph.finalize()	# avoid modifications,read only

    def _build_model(self):

        l_input = Input( batch_shape=(None, NUM_STATE) )
        l_dense = Dense(16, activation='relu')(l_input)

        out_actions = Dense(NUM_ACTIONS, activation='softmax')(l_dense)
        out_value   = Dense(1, activation='linear')(l_dense)

        model = Model(inputs=[l_input], outputs=[out_actions, out_value])
        model._make_predict_function()	# have to initialize before threading

        return model

    def _build_graph(self, model):
        s_t = tf.placeholder(tf.float32, shape=(None, NUM_STATE))
        a_t = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
        r_t = tf.placeholder(tf.float32, shape=(None, 1)) # not immediate, but discounted n step reward

        p, v = model(s_t)	#output policy, value

        log_prob = tf.log( tf.reduce_sum(p * a_t, axis=1, keep_dims=True) + 1e-10)
        advantage = r_t - v

        loss_policy = - log_prob * tf.stop_gradient(advantage)									# maximize policy
        loss_value  = LOSS_V * tf.square(advantage)												# minimize value error
        entropy = LOSS_ENTROPY * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1, keep_dims=True)	# maximize entropy (regularization)   improve exploration by limiting the premature convergence to suboptimal policy

        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99)
        minimize = optimizer.minimize(loss_total)	#minize the loss function

        return s_t, a_t, r_t, minimize

    def optimize(self):
        if len(self.train_queue[0]) < MIN_BATCH:
            time.sleep(0)	# yield, get more examples from agents
            return

        with self.lock_queue:
            if len(self.train_queue[0]) < MIN_BATCH:	# more thread could have passed without lock
                return 									# we can't yield inside lock

            s, a, r, s_, s_mask = self.train_queue
            self.train_queue = [ [], [], [], [], [] ]

        s = np.vstack(s)
        a = np.vstack(a)
        r = np.vstack(r)
        s_ = np.vstack(s_)
        s_mask = np.vstack(s_mask)

        if len(s) > 5*MIN_BATCH: print("Optimizer alert! Minimizing batch of %d" % len(s))

        v = self.predict_v(s_)	#input state s_ to get value
        r = r + GAMMA_N * v * s_mask	# set v to 0 where s_ is terminal state

        s_t, a_t, r_t, minimize = self.graph
        self.session.run(minimize, feed_dict={s_t: s, a_t: a, r_t: r})

    def train_push(self, s, a, r, s_):  #push example into train_queue,r= n-step-rewards
        with self.lock_queue:
            self.train_queue[0].append(s)
            self.train_queue[1].append(a)
            self.train_queue[2].append(r)

            if s_ is None:
                self.train_queue[3].append(NONE_STATE)
                self.train_queue[4].append(0.)
            else:
                self.train_queue[3].append(s_)
                self.train_queue[4].append(1.)

    def predict(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p, v

    def predict_p(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p

    def predict_v(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return v

#---------
frames = 0	#
#define Agent class(with threads):
class Agent:
    def __init__(self, eps_start, eps_end, eps_steps):
        self.eps_start = eps_start
        self.eps_end   = eps_end
        self.eps_steps = eps_steps

        self.memory = []	# used for n_step return
        self.R = 0. #reward

    def getEpsilon(self):   #get explore rate, come down from 0.4 to 0.15 during 75000 steps
        if(frames >= self.eps_steps):
            return self.eps_end
        else:
            return self.eps_start + frames * (self.eps_end - self.eps_start) / self.eps_steps	# linearly interpolate

    def act(self, s):   #choose action
        eps = self.getEpsilon()
        global frames; frames = frames + 1

        if random.random() < eps:
            return random.randint(0, NUM_ACTIONS-1)

        else:
            s = np.array([s])
            p = brain.predict_p(s)[0]
            #print("policy distribute",p)
            a = np.random.choice(NUM_ACTIONS, p=p)  #random choose action according to distribution p

            return a

    def train(self, s, a, r, s_):   #
        def get_sample(memory, n):  #compute n-step discounted reward and return a proper tuple
            s, a, _, _  = memory[0]     #first state s0 and the action a0
            _, _, _, s_ = memory[n-1]   #last state s(n)

            return s, a, self.R, s_

        a_cats = np.zeros(NUM_ACTIONS)	# turn action into one-hot representation
        a_cats[a] = 1

        self.memory.append( (s, a_cats, r, s_) )    #stores the current transition in memory, which is used to compute the n-step return

        #print("R1:",self.R)

        self.R = ( self.R + r * GAMMA_N ) / GAMMA   #more effective way to compute n-step rewards
        #print("R2:", self.R)

        if s_ is None:  #don't have next state because done==True
            while len(self.memory) > 0: #In loop, we shorten the buffer in each iteration and compute the n-step return, where n is equal to the current length of the buffer.
                n = len(self.memory)
                s, a, r, s_ = get_sample(self.memory, n)
                brain.train_push(s, a, r, s_)

                self.R = ( self.R - self.memory[0][2] ) / GAMMA
                self.memory.pop(0)

            self.R = 0

        #Last n samples are stored in this buffer and when there are enough of them, n-step discounted reward R is computed, a tuple (s_0, a_0, R, s_n) is inserted into the brain’s training queue
        if len(self.memory) >= N_STEP_RETURN:
            s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
            brain.train_push(s, a, r, s_)

            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)  #delete the first element in memory

        #print("rewards now:",self.R)

    # possible edge case - if an episode ends in <N steps, the computation is incorrect

#---------
class Environment(threading.Thread):
    stop_signal = False

    def __init__(self, render=False, eps_start=EPS_START, eps_end=EPS_STOP, eps_steps=EPS_STEPS):
        threading.Thread.__init__(self)

        self.render = render
        self.env = gym.make(ENV)
        self.agent = Agent(eps_start, eps_end, eps_steps)

    def runEpisode(self):   #combine brain's and agent's methods together
        s = self.env.reset()

        R = 0
        times = 0
        while True:
            time.sleep(THREAD_DELAY) # yield  suspends execution for the given number of seconds

            if self.render: self.env.render()   #rend environment

            a = self.agent.act(s)   #choose action a
            s_, r, done, info = self.env.step(a)    #get next state, reward, is done from environment

            if done: # terminal state
                s_ = None

            self.agent.train(s, a, r, s_)   #call agent train method

            s = s_
            R += r
            times += 1
            #R += r  #

            if done or self.stop_signal:
                break

        print("Total R:", R)
        #print("times without done:", times)

    def run(self):
        while not self.stop_signal:
            self.runEpisode()

    def stop(self):
        self.stop_signal = True

#---------
class Optimizer(threading.Thread):
    stop_signal = False

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        while not self.stop_signal:
            brain.optimize()

    def stop(self):
        self.stop_signal = True

#-- main
env_test = Environment(render=True, eps_start=0., eps_end=0.)
NUM_STATE = env_test.env.observation_space.shape[0]
NUM_ACTIONS = env_test.env.action_space.n
NONE_STATE = np.zeros(NUM_STATE)

brain = Brain()	# brain is global in A3C

envs = [Environment() for i in range(THREADS)]
opts = [Optimizer() for i in range(OPTIMIZERS)]



for o in opts:
    o.start()

for e in envs:
    e.start()


print(time.ctime())
time.sleep(RUN_TIME)    #run RUN_TIME seconds
#print(time.ctime())

for e in envs:
    e.stop()
    print(e.is_alive())
for e in envs:
    e.join()
    print(e.is_alive())


for o in opts:
    o.stop()
for o in opts:
    o.join()

#print(time.ctime())


#print(time.ctime())


print("Training finished")
time.sleep(2)
env_test.run()  #run the environment with trained data
