import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)

class PlaneBallEnv(gym.Env):
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
        self.ball_radius = 0.1
        self.force = 10.0
        self.tau = 0.02  # seconds between state updates

        # Maximum values for observation
        self.ball_x = 1
        self.ball_y = 1
        self.x_Aplha = 0.5*np.pi
        self.y_Beta = 0.5*np.pi
        self.Alpha_vel = 8
        self.Beta_vel = 8
        self.ball_vel = np.finfo(np.float32).max


        high = np.array([
            self.x_Aplha,
            self.Alpha_vel,
            self.y_Beta,
            self.Beta_vel,
            self.ball_x,
            self.ball_y,
            self.ball_vel])

        #Action space type: Box(2), Torque around x and y axises.
        self.action_space = spaces.Box(np.array([-self.max_torque,-self.max_torque]), np.array([self.max_torque,self.max_torque]))

        #Observation space type: Box(7), Alpha, Beta and related angle velocity of plane, ball position and velocity.
        self.observation_space = spaces.Box(-high, high)

        self._seed()
        self._reset()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        action_X = np.clip(action, -self.max_torque, self.max_torque)[0]
        action_Y = np.clip(action, -self.max_torque, self.max_torque)[1]
        state = self.state
        x_alpha, alpha_vel, y_beta, beta_vel, ball_x, ball_y, ball_vel = state

        #degree to radian
        #x_alpha_rad = x_alpha/2*np.pi
        #y_beta_rad = y_beta/2*np.pi


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
                reward = 1.0
            else:
                reward = -1.0
        else:
            reward = -100.0

        return np.array(self.state), reward, done, {}

    def _reset(self):

        self.state = np.array([self.np_random.uniform(low=-self.x_Aplha, high=self.x_Aplha), 0, self.np_random.uniform(low=-self.y_Beta, high=self.y_Beta), 0, self.np_random.uniform(low=-self.ball_x, high=self.ball_x), self.np_random.uniform(low=-self.ball_y, high=self.ball_y), 0])

        return np.array(self.state)

    '''
    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = 2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
        '''
