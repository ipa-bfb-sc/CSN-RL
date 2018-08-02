"""
Modified from Morvan python: https://morvanzhou.github.io/tutorials

Distributing workers in parallel to collect data, then stop worker's roll-out and train PPO on collected data.
Restart workers once PPO is updated.

The global PPO updating rule is adopted from DeepMind's paper (DPPO):
Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]

"""
import os
import timeit
import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym, threading, queue
import environments
from datetime import datetime
timenow = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


ST_MAX = 300000
EP_MAX = 1000
EP_LEN = 200

#GAMMA = 0.9                 # reward discount factor
#A_LR = 0.00001               # learning rate for actor
#C_LR = 0.0002               # learning rate for critic
#MIN_BATCH_SIZE = 100         # minimum batch size for updating PPO
#UPDATE_STEP = 10            # loop update operation n-steps
#EPSILON = 0.2               # for clipping surrogate objective


# state and action dimension
#S_DIM, A_DIM = 4, 1      # CartPole
#S_DIM, A_DIM = 7, 2     # PlaneBall
#S_DIM, A_DIM = 20, 1     # CirTyrtleBot


def save_data(filepath, data):
    path = os.path.join('save', filepath)
    with open(path, 'w') as f:
        json.dump(data, f)


class PPO(object):
    def __init__(self, s_dim, a_dim, actor_lr, critic_lr, update_steps, eps):
        self.sess = tf.Session()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.update_steps = update_steps
        self.epsilon = eps
        self.tfs = tf.placeholder(tf.float32, [None, self.s_dim], 'state')


        # critic
        # critic network
        l1 = tf.layers.dense(self.tfs, 24, tf.nn.relu)
        nn = tf.layers.dense(l1, 24, tf.nn.relu)
        nn = tf.layers.dense(nn, 24, tf.nn.relu)
        self.v = tf.layers.dense(nn, 1)

        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.advantage = self.tfdc_r - self.v
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        self.ctrain_op = tf.train.AdamOptimizer(self.critic_lr).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        self.sample_op = tf.squeeze(pi.sample(1), axis=0)  # operation of choosing action
        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, a_dim], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')

        # Using clipped surrogate objective

        ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
        #ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)
        surr = ratio * self.tfadv                       # surrogate loss

        self.aloss = -tf.reduce_mean(tf.minimum(
            surr,
            tf.clip_by_value(ratio, 1. - self.epsilon, 1. + self.epsilon) * self.tfadv))

        self.atrain_op = tf.train.AdamOptimizer(self.actor_lr).minimize(self.aloss)
        self.sess.run(tf.global_variables_initializer())

    def update(self):
        global GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            #if GLOBAL_EP < EP_MAX:
            if GLOBAL_STEPS < ST_MAX:
                UPDATE_EVENT.wait()                     # wait until get batch of data
                self.sess.run(self.update_oldpi_op)     # copy pi to old pi
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]      # collect data from all workers
                #print("QUEUE.qsize()={}".format(len(data)))
                #print('data:'.format(data))
                data = np.vstack(data)

                s, a, r = data[:, :self.s_dim], data[:, self.s_dim: self.s_dim + self.a_dim], data[:, -1:]
                adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
                # update actor and critic in a update loop
                [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(self.update_steps)]
                [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(self.update_steps)]
                UPDATE_EVENT.clear()        # updating finished
                GLOBAL_UPDATE_COUNTER = 0   # reset counter
                ROLLING_EVENT.set()         # set roll-out available

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 24, tf.nn.relu, trainable=trainable)
            l1 = tf.layers.dense(l1, 24, tf.nn.relu, trainable=trainable)
            l1 = tf.layers.dense(l1, 24, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, self.a_dim, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, self.a_dim, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -1, 1)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]


class Worker(object):
    def __init__(self, wid, env, ppo, gamma, batch_size):
        self.wid = wid
        self.env = gym.make(env).unwrapped
        self.ppo = ppo
        self.gamma = gamma
        self.batch_size = batch_size

    def work(self):
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER, GLOBAL_STEPS
        ep=0
        while not COORD.should_stop():
            ep+=1
            s = self.env.reset()
            ep_r = 0
            buffer_s, buffer_a, buffer_r = [], [], []
            for t in range(EP_LEN):
                duration_begin = timeit.default_timer()
                if not ROLLING_EVENT.is_set():                  # while global PPO is updating
                    ROLLING_EVENT.wait()                        # wait until PPO is updated
                    buffer_s, buffer_a, buffer_r = [], [], []   # clear history buffer, use new policy to collect data
                a = self.ppo.choose_action(s)
                s_, r, done, _ = self.env.step(a)
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r + 8) / 8)                    # normalize reward, find to be useful
                s = s_
                ep_r += r

                GLOBAL_STEPS += 1
                GLOBAL_UPDATE_COUNTER += 1               # count to minimum batch size, no need to wait other workers
                if t == EP_LEN - 1 or GLOBAL_UPDATE_COUNTER >= self.batch_size:
                    v_s_ = self.ppo.get_v(s_)
                    discounted_r = []                           # compute discounted reward
                    for r in buffer_r[::-1]:
                        v_s_ = r + self.gamma * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    # [[s1_a,s1_b,s1_c,s1_d],[s2_a,s2_b,s2_c,s2_d],...] to [[s1_a s1_b s1_c s1_d]
                    #                                                       [s2_a s2_b s2_c s2_d],...]
                    # [[a1],[a2],[a3],...] to [[a1]
                    #                          [a2]
                    #                          [a3]...]
                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    #print('action-vstack:{}'.format(ba))

                    buffer_s, buffer_a, buffer_r = [], [], []
                    QUEUE.put(np.hstack((bs, ba, br)))          # put data in the queue, data.form=[[s1_a s1_b s1_c s1_d a1 r1]...]
                    #print('hstack:{}'.format(np.hstack((bs, ba, br))))
                    #print('hstack-len:{}'.format(len(np.hstack((bs, ba, br)))))


                    if GLOBAL_UPDATE_COUNTER >= self.batch_size:
                        ROLLING_EVENT.clear()       # stop collecting data
                        UPDATE_EVENT.set()          # globalPPO update

                    if GLOBAL_STEPS >= ST_MAX:         # stop training
                    #if abs(GLOBAL_STEPS - ST_MAX) < 400:

                        COORD.request_stop()
                        #break
                #CSN
                #print('GLOBAL_UPDATE_COUNTER:{}'.format(GLOBAL_UPDATE_COUNTER))
                if done:
                    break
            GLOBAL_EP += 1
            duration = timeit.default_timer() - duration_begin
            # record reward changes, plot later
            history['reward'].append(ep_r)
            history['steps'].append(GLOBAL_STEPS)
            history['episode'].append(GLOBAL_EP)
            history['ep_steps'].append(t)
            history['duration'].append(duration)
            if len(GLOBAL_RUNNING_R) == 0: GLOBAL_RUNNING_R.append(ep_r)
            else: GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1]*self.gamma + ep_r*(1-self.gamma))
            history['discount_r']=GLOBAL_RUNNING_R
            #print('{0:.1f}%'.format(GLOBAL_EP/EP_MAX*100), '|W%i' % self.wid,  '|Ep_r: %.2f' % ep_r,)
            print('|W%i' % self.wid, 'steps:{}'.format(GLOBAL_STEPS),'episode:{}'.format(GLOBAL_EP), 'episode steps:{}'.format(t+1),  '|Ep_r: %.2f' % ep_r, 'duration:{}'.format(duration))
            save_data('history5_{}'.format(timenow), history)

            #DURATION += duration
            #print('d:{}'.format(DURATION))
            #print('worker id:{}, step:{}, episode:{}, episode steps:{}, episode reward:{}'.format(self.wid, GLOBAL_STEPS, GLOBAL_EP, t, ep_r))

        #save_data('history5_{}'.format(timenow), history)


def test(game, render):
    env = gym.make(game)
    for t in range(30):
        s = env.reset()
        epr = 0
        for i in range(EP_LEN):
            if render:
                env.render()
            action = GLOBAL_PPO.choose_action(s)
            s_, r, done, _ = env.step(action)
            s = s_
            epr += r
            if done:
                break
        print('episode:{}, ep_steps:{}, ep_reward:{}'.format(t, i, epr))



if __name__ == '__main__':

    d_begin = timeit.default_timer()

    GAME = 'ContinuousCartPole-v0'
    #GAME = 'PlaneBall-v0'
    # GAME = 'GazeboCircuit2TurtlebotLidarNn-v1'

    N_WORKER = 4  # parallel workers

    GLOBAL_PPO = PPO(s_dim=4, a_dim=1, actor_lr=0.0001, critic_lr=0.00002, update_steps=10, eps=0.2)

    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    UPDATE_EVENT.clear()            # not update now
    ROLLING_EVENT.set()             # start to roll out
    workers = [Worker(wid=i, env=GAME, ppo=GLOBAL_PPO, gamma=0.99, batch_size=32) for i in range(N_WORKER)]
    
    GLOBAL_UPDATE_COUNTER, GLOBAL_EP, GLOBAL_STEPS= 0, 0, 0
    GLOBAL_RUNNING_R = []
    history={'episode':[],'steps':[],'reward':[],'discount_r':[], 'ep_steps':[], 'duration':[]}


    COORD = tf.train.Coordinator()
    QUEUE = queue.Queue()           # workers putting data in this queue
    threads = []
    for worker in workers:          # worker threads
        t = threading.Thread(target=worker.work, args=())
        t.start()                   # training
        threads.append(t)
    # add a PPO updating thread
    threads.append(threading.Thread(target=GLOBAL_PPO.update,))
    threads[-1].start()
    COORD.join(threads)

    d = timeit.default_timer() - d_begin
    history['duration_total'] = d
    save_data('history5_{}'.format(timenow), history)
    print('dd:{}'.format(history['duration_total']))

    test(GAME, False)

