# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import torch
import torch.nn as nn
import random
import math

import numpy as np
# import mindspore as ms
# from mindspore.common import dtype as mstype
# from mindspore import ops
# from mindspore.ops import operations as P
#
# from models.cflownets.loss_network import CriticTrainNetWrapper
# from models.cflownets.config import CONFIG as cfg
# from models.cflownets.utils import select_action_base_probability, softmax_matrix


#
class FlowModel(nn.Module):
    def __init__(self, num_inputs, hidden_dim, num_cond_inputs=None):
        super(FlowModel, self).__init__()
        # Edge flow network architecture
        self.c_l1 = nn.Linear(num_inputs, hidden_dim, bias=True)
        self.c_l2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.c_l3 = nn.Linear(hidden_dim, 1, bias=True)
        self.c_relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, sa):
        q1 = self.c_relu(self.c_l1(sa))
        q1 = self.c_relu(self.c_l2(q1))
        output = self.softplus(self.c_l3(q1))
        return output

    def log_probs(self, inputs, cond_inputs=None):

        inputs = self.softplus(self.c_l3(self.c_relu(self.c_l2(self.c_relu(self.c_l1(inputs))))))

        log_probs = (-0.5 * inputs.pow(2) - 0.5 * math.log(2 * math.pi)).sum(
            -1, keepdim=True)
        return log_probs.sum(-1, keepdim=True)

    def sample(self, num_samples=None, noise=None, cond_inputs=None):
        if noise is None:
            noise = torch.Tensor(num_samples, self.num_inputs).normal_(mean=0, std=0.01)
        device = next(self.parameters()).device
        noise = noise.to(device)
        if cond_inputs is not None:
            cond_inputs = cond_inputs.to(device)
        samples = self.mlp(noise)
        return samples

#   def forward(self, x):
#     logits = self.mlp(x)
#
#     P_F = logits[..., self.dim:]
#     P_B = logits[..., :self.dim]
#     return P_F, P_B
#
# class CFN:
#     def __init__(self, state_dim, action_dim, hidden_dim, max_action, uniform_action_size, density_model,
#                  discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
#         self.density_model = density_model
#         self.max_action = max_action
#         self.discount = discount
#         self.tau = tau
#         self.policy_noise = policy_noise
#         self.noise_clip = noise_clip
#         self.policy_freq = policy_freq
#         self.uniform_action_size = uniform_action_size
#         self.uniform_action = np.random.uniform(low=-max_action, high=max_action,
#                                                 size=(uniform_action_size, action_dim))
#         self.total_it = 0
#         self.action_dim = action_dim
#         self.state_dim = state_dim
#
#     def select_action(self, state, is_max):
#         sample_action = np.random.uniform(low=-self.max_action, high=self.max_action, size=(1000, self.action_dim))
#         self.critic.set_train(False)
#         state = np.repeat(state.reshape(1, -1), 1000, axis=0)
#         sa = np.concatenate((state, sample_action), axis=-1)
#         sa = Tensor(sa, mstype.float32)
#         edge_flow = self.critic(sa).reshape(-1)
#         edge_flow = edge_flow.asnumpy()
#         edge_flow_norm = np.array(softmax_matrix(edge_flow))
#         if is_max == 0:
#             action = select_action_base_probability(sample_action, edge_flow_norm)
#         elif is_max == 1:
#             action = sample_action[edge_flow.argmax()]
#         return action
#
#     def cal_inflow_sa(self, next_state, state, action, batch_size, max_episode_steps, sample_flow_num):
#         """
#         calculate inflow state and action
#         """
#         uniform_action = np.random.uniform(low=-self.max_action, high=self.max_action,
#                                            size=(batch_size, max_episode_steps, sample_flow_num, self.action_dim))
#         current_state = np.repeat(next_state, sample_flow_num, axis=2).reshape(
#             batch_size, max_episode_steps, sample_flow_num, -1)
#         cat_state_action = np.concatenate((current_state, uniform_action), axis=-1)
#         cat_state_action = Tensor(cat_state_action, mstype.float32)
#         inflow_state = self.transaction(cat_state_action)
#         state_ms = Tensor(state.reshape(batch_size, max_episode_steps, -1, self.state_dim), mstype.float32)
#         inflow_state = ops.concat([inflow_state, state_ms], axis=2)
#         inflow_action = np.concatenate((uniform_action, action.reshape(
#             batch_size, max_episode_steps, -1, self.action_dim)), axis=2)
#         inflow_action = Tensor(inflow_action, mstype.float32)
#         return inflow_state, inflow_action
#
#     def cal_outflow_sa(self, next_state, action, batch_size, max_episode_steps, sample_flow_num):
#         """
#         calculate outflow state and action
#         """
#         uniform_action = np.random.uniform(low=-self.max_action, high=self.max_action,
#                                            size=(batch_size, max_episode_steps, sample_flow_num, self.action_dim))
#         outflow_state = np.repeat(next_state, sample_flow_num + 1, axis=2).reshape(
#             batch_size, max_episode_steps, sample_flow_num + 1, -1)
#         last_action = np.zeros((batch_size, 1, 1))
#         last_action = np.concatenate((action[:, 1:, :], last_action), axis=1)
#         last_action_ = last_action.reshape((batch_size, max_episode_steps, -1, self.action_dim))
#         outflow_action = np.concatenate((uniform_action, last_action_), axis=2)
#         return outflow_state, outflow_action
#
#     def train(self, state, action, reward, next_state, density_model_train_net, not_done, done_true):
#         sample_flow_num = cfg['sample_flow_num']
#         batch_size = cfg['batch_size']
#         max_episode_steps = cfg['max_episode_steps']
#
#         # in flow
#         inflow_state, inflow_action = self.cal_inflow_sa(next_state, state,
#                                                          action, batch_size, max_episode_steps, sample_flow_num)
#
#         # out flow
#         outflow_state, outflow_action = self.cal_outflow_sa(next_state,
#                                                             action, batch_size, max_episode_steps, sample_flow_num)
#
#         density_model_train_net.set_train()
#
#         outflow_state = Tensor(outflow_state, mstype.float32)
#         outflow_action = Tensor(outflow_action, mstype.float32)
#         reward = Tensor(reward, mstype.float32)
#         critic_loss = density_model_train_net(inflow_state, inflow_action, outflow_state, outflow_action, not_done,
#                                               done_true, reward)
#         print('every critic_loss:', critic_loss)
#         return critic_loss
