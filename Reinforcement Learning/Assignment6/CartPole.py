import gym
import math
import numpy as np
from gym.utils import seeding
from gym import spaces, logger
import matplotlib.pyplot as plt


g =  9.81
class CartPoleEnv(gym.Env):

    def __init__(self):
        self.masscart = 6.0
        self.masspole = 3.0
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.8 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 100.0
        self.timeStep = 0.01  # seconds between state updates

        # theta at which the episode is terminated
        self.theta_terminal = 57.2958 * 2 * math.pi / 360  # 1 rad = 57.2958 degree
        self.theta_target = 2.864789 * 2 * math.pi / 360
        self.x_terminal = 5
        self.x_target = 0.1

        high = np.array([
            self.x_terminal * 2,
            np.finfo(np.float32).max,
            self.theta_terminal * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_terminal = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def getForce(self, x, x_dot, theta, theta_dot):
        k = np.random.uniform(0, 1, 4)
        return min(100, max(-100, k[0]*x + k[1]*x_dot + k[2]*theta + k[3]*theta_dot))

    def addNoise(self):
        mu = np.array([0., 0., 0., 0.])
        cov = np.array([[0.004, 0., 0., 0.],[0., 0.04, 0., 0.], [0., 0., 0.001, 0.], [0., 0., 0., 0.01]])
        noise = np.random.multivariate_normal(mu, cov, 1)
        return np.add(self.state, noise)
    
    def getNextState(self, action):
        state = self.addNoise()
        x, x_dot, theta, theta_dot = state.reshape(4)
        force = self.getForce(x, x_dot, theta, theta_dot)
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (g * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x  = x + self.timeStep * x_dot
        x_dot = x_dot + self.timeStep * xacc
        theta = theta + self.timeStep * theta_dot
        theta_dot = theta_dot + self.timeStep * thetaacc

        self.state = (x, x_dot, theta, theta_dot)
        
        self.checkIfTerminal()
        self.checkIfInTarget()
    
    def checkIfTerminal(self):
        x = self.state[0]
        theta = self.state[2]
        self.isTerminal =  x < -self.x_terminal \
                      or x > self.x_terminal \
                      or theta < -self.theta_terminal \
                      or theta > self.theta_terminal
        self.isTerminal = bool(self.isTerminal)
    
    def checkIfInTarget(self):
        x = self.state[0]
        theta = self.state[2]
        self.isInTarget = x < -self.x_target \
                     or x > self.x_target \
                     or theta < -self.theta_target \
                     or theta > self.theta_target
        self.isInTarget = bool(self.isInTarget)
        
    def getReward(self):
        if not self.isTerminal and self.isInTarget:
            # is not terminated and in target region
            reward = 0.0
        else:
            reward = -1.0
        return reward

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_terminal*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
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

            self._pole_geom = pole

        if self.state is None: return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
        pole.v = [(l,b), (l,t), (r,t), (r,b)]

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
    
    def state_tracking(self, x, x_dot, theta, theta_dot):

        # force set by 0
        force = 0

        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (g * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x  = x + self.timeStep * x_dot
        x_dot = x_dot + self.timeStep * xacc
        theta = theta + self.timeStep * theta_dot
        theta_dot = theta_dot + self.timeStep * thetaacc

        state = (x, x_dot, theta, theta_dot)
        return state

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_terminal = None
