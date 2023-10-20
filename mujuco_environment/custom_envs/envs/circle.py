import gc
import math

import gym
import numpy as np
from matplotlib import pyplot as plt


class CircleEnv(gym.Env):
    # -------------------------
    # Constructor
    # -------------------------
    def __init__(self, speed=0.02, sigma1=0.01, sigma2=0.03, max_step=512, **_kwargs):
        self.state = np.zeros((5, 2), dtype=np.float32)
        self.max_step = max_step
        self.n_step = 1
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.done = False
        self.use_render = False
        self.color = [np.array([1.0, 0.0, 0.0])]
        self.p = [0, 0]
        self.xs = []
        self.ys = []
        self.speed = speed
        self.observation_space = gym.spaces.Box(low=np.ones([10], dtype=np.float32) * -1,
                                                high=np.ones([10], dtype=np.float32))
        self.action_space = gym.spaces.Box(low=np.ones([2], dtype=np.float32) * float('-inf'),
                                           high=np.ones([2], dtype=np.float32) * float('inf'))

    # self.fig = plt.figure()
    # self.ax  = self.fig.add_subplot(111)
    # mng = plt.get_current_fig_manager()
    # mng.resize(*mng.window.maxsize())

    # -------------------------
    # Step
    # -------------------------
    def step(self, action: list):
        norm = math.sqrt(action[0] * action[0] + action[1] * action[1])

        if norm > 1e-8:
            action[0] /= norm
            action[1] /= norm
        else:
            action[0] = 1.0
            action[1] = 0.0

        self.p[0] += action[0] * self.speed + self.sigma2 * np.random.randn()
        self.p[1] += action[1] * self.speed + self.sigma2 * np.random.randn()
        self.xs.append(self.p[0])
        self.ys.append(self.p[1])
        self.n_step += 1
        # noise = np.random.normal(0, self.var, 2)
        for i in range(4):
            self.state[i, :] = self.state[i + 1, :]
        self.state[4, :] = np.array(self.p)

        reward = 0.0

        # if self.done:
        #     fig = plt.figure()
        #     self.ax = fig.add_subplot(111)
        #     self.ax.cla()
        #     self.ax.set_aspect(aspect=1.0)

        # if self.n_step >= 129 or abs(self.p[0]) >= 1 or abs(self.p[1]) >= 1:
        if self.n_step >= 1024:
            self.done = True
        else:
            self.done = False

        return np.copy(self.state.flatten()), reward, self.done, {}

    # -------------------------
    # Reset
    # -------------------------
    def reset(self, start=(0.0, 0.0)):
        if self.use_render:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            self.ax.cla()
            self.ax.set_aspect(aspect=1.0)
            self.render(mode='reset')
        # self.p[0]   = self.sigma1 * np.random.randn()
        # self.p[1]   = self.sigma1 * np.random.randn()
        self.p[0] = start[0]
        self.p[1] = start[1]
        del self.xs, self.ys
        self.xs = [self.p[0]]
        self.ys = [self.p[1]]
        self.n_step = 1

        for i in range(5):
            self.state[i, 0] = self.p[0]
            self.state[i, 1] = self.p[1]

        self.use_render = False
        gc.collect()
        return np.copy(self.state.flatten())

    # -------------------------
    # Render
    # -------------------------
    def render(self, mode="human"):
        if self.done and mode == 'reset':
            self.ax.scatter(self.xs, self.ys, s=4)
            plt.grid()
            plt.xlim(-1, 1)
            plt.ylim(-1, 1)
        self.use_render = True
        # return 123
        # else:
        #     pass
        # plt.pause(0.002)

    # -------------------------
    # Get all points
    # -------------------------
    def get_points(self):
        return self.xs, self.ys

    # -------------------------
    # Get all points
    # -------------------------
    def get_image(self):
        print("abc")
        return None


    # -------------------------
    # Close the environment
    # -------------------------
    def close(self):
        pass


# -------------------------
# Make an environment
# -------------------------
def make(max_step=512):
    return CircleEnv(max_step=max_step)
