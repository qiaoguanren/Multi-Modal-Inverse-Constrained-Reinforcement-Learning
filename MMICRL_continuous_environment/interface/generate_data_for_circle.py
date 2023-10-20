import math
import os

import numpy as np
import pickle as pkl
# from mujuco_environment.custom_envs.envs.circle import plt
from matplotlib import pyplot as plt
from tqdm import tqdm


# -----------------------
# Circle equation
# -----------------------
def circle(theta, r):
    return r * math.cos(theta), r * math.sin(theta)


# -----------------------
# Expert policy step
# -----------------------
def policy_step(p, r, up=True, clockwise=True):
    if up:
        offset = r
    else:
        offset = -r

    if clockwise:
        theta = math.atan2(p[1] - offset, p[0]) + 2 * math.asin(0.005 / r)
    else:
        theta = math.atan2(p[1] - offset, p[0]) - 2 * math.asin(0.005 / r)

    p_tar = circle(theta, r)
    dx = p_tar[0] - p[0]
    dy = p_tar[1] - p[1] + offset
    norm = math.sqrt(dx * dx + dy * dy)

    return [dx / norm, dy / norm] if norm > 1e-8 else [0, 0]


# -----------------------
# Main
# -----------------------
import gym
import mujuco_environment.custom_envs

env = gym.make(id='Circle-v0', **env_configs_copy)
std = str(5e-3)
rs = [0.4, 0.2, 0.2]
cs = [True, True, False]
us = [True, True, False]
idx = 0

for i in range(len(rs)):
    print("r = {:.2f}".format(rs[i]))
    save_circle_data_path = '../data/expert_data/Circle-v0-std-{0}-id-{1}/'.format(std, i)
    if not os.path.exists(save_circle_data_path):
        os.mkdir(save_circle_data_path)
    for j in tqdm(range(2)):  # default 50
        state = env.reset()
        # s_traj.append([])
        # a_traj.append([])
        # c_traj.append([])
        s_traj = []
        a_traj = []
        r_traj = []
        c_traj = []

        while True:
            action = policy_step(state[-2:], rs[i], up=us[i], clockwise=cs[i])
            s_traj.append(state)
            a_traj.append(action)
            r_traj.append(0)
            one_hot_code = np.zeros(len(rs))
            one_hot_code[i] = 1
            c_traj.append(one_hot_code)

            state, reward, done, info = env.step(action)
            env.render()

            if done:
                s_traj = np.array(s_traj, dtype=np.float32)
                a_traj = np.array(a_traj, dtype=np.float32)
                c_traj = np.array(c_traj, dtype=int)
                idx += 1
                plt.savefig('tmp{0}.png'.format(str(one_hot_code)))
                break
        # pkl.dump({'original_observations': s_traj,
        #           'actions': a_traj,
        #           'codes': c_traj,
        #           'reward': r_traj,
        #           'reward_sum': 0,
        #           },
        #          open(save_circle_data_path +
        #               "scene-{1}_len-{3}.pkl".format(
        #                   std,
        #                   idx,
        #                   str(one_hot_code),
        #                   len(s_traj)
        #               ),
        #               "wb"))
