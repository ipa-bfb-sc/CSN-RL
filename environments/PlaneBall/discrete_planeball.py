import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import bisect



logger = logging.getLogger(__name__)

class DiscretePlaneBallEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.massplane = 1.0
        self.massball = 0.1
        self.total_mass = (self.massball + self.massplane)
        self.length = 2
        self.max_torque = 2.
        self.ball_radius = 0.2
        self.force = 10.0
        self.tau = 0.02  # seconds between state updates

        # Maximum values for observation
        self.ball_x = 1
        self.ball_y = 1
        self.x_Aplha = 0.5*np.pi # rad
        self.y_Beta = 0.5*np.pi # rad
        self.Alpha_vel = 8 # rad/s
        self.Beta_vel = 8 # rad/s
        self.ball_vel = np.finfo(np.float32).max


        high = np.array([
            self.x_Aplha,
            self.Alpha_vel,
            self.y_Beta,
            self.Beta_vel,
            self.ball_x,
            self.ball_y,
            self.ball_vel])

        #Action space type: Discrete(81), Actions around x and y axises from interval[-2,2].
        self.action_space = spaces.Discrete(81)
        #Observation space type: Box(7), Alpha, Beta and related angle velocity of plane, ball position and velocity.
        self.observation_space = spaces.Box(-high, high)

        self._seed()
        self._reset()
        self.viewer = None
        self.state = None

        #self.steps_beyond_done = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))


        #alias action to torque on X and Y
        x, y = np.meshgrid(np.linspace(-2., 2., 9), np.linspace(-2., 2., 9))
        ac = np.rec.fromarrays([y, x])
        ac2 = ac.flatten()
        torque_xy = list(ac2[action])


        action_X = torque_xy[0]
        action_Y = torque_xy[1]
        state = self.state
        x_alpha, alpha_vel, y_beta, beta_vel, ball_x, ball_y, ball_vel = state

        #moment of inertia
        I = 5/12*(self.massplane*self.length**2)

        #angular acceleration
        X_acce = action_X/I
        Y_acce = action_Y/I

        # Angular velocity = old angular velocity + Angular acceleration* time
        alpha_vel = alpha_vel + X_acce * self.tau
        beta_vel = beta_vel + Y_acce * self.tau

        x_alpha = x_alpha + alpha_vel * self.tau
        y_beta = y_beta + beta_vel * self.tau

        alpha_vel = np.clip(alpha_vel, -8, 8)
        beta_vel = np.clip(beta_vel, -8, 8)

        #####
        ball_vel_x = alpha_vel * ball_x
        ball_vel_y = beta_vel * ball_y

        ball_vel = (ball_vel_x**2 + ball_vel_y**2)**0.5

        ball_x = ball_x + ball_vel_x * self.tau
        ball_y = ball_y + ball_vel_y * self.tau


        self.state = np.array([x_alpha, alpha_vel, y_beta, beta_vel, ball_x, ball_y, ball_vel])


        done =  ball_x < -0.5*self.length \
                or ball_x > 0.5*self.length \
                or ball_y < -0.5 * self.length \
                or ball_y > 0.5 * self.length



        done = bool(done)

        if not done:
            if -self.ball_radius <= ball_x <= self.ball_radius and -self.ball_radius <= ball_y <= self.ball_radius:
                reward = 100
            else:
                reward = -1.0

        else:
            reward = -100

        return np.array(self.state), reward, done, {}

    def _reset(self):
        '''
        high = np.array([
            self.x_Aplha,
            self.Alpha_vel,
            self.y_Beta,
            self.Beta_vel,
            self.ball_x,
            self.ball_y,
            self.ball_vel])
        self.state = self.np_random.uniform(low=-high, high=high, size=(7,))
        '''
        self.state = np.array([self.np_random.uniform(low=-self.x_Aplha, high=self.x_Aplha), 0, self.np_random.uniform(low=-self.y_Beta, high=self.y_Beta), 0, self.np_random.uniform(low=-self.ball_x, high=self.ball_x), self.np_random.uniform(low=-self.ball_y, high=self.ball_y), 0])
        #self.steps_beyond_done = None
        return np.array(self.state)

