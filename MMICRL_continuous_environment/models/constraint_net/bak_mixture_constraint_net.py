import os
import random
from itertools import accumulate
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union, List
import numpy as np
import torch as th
import torch.nn.functional as F
from models.constraint_net.constraint_net import ConstraintNet
from cirl_stable_baselines3.common.torch_layers import create_mlp
from cirl_stable_baselines3.common.utils import update_learning_rate
from torch import nn
from tqdm import tqdm

from utils.data_utils import build_rnn_input
from utils.model_utils import build_code


class MixtureConstraintNet(ConstraintNet):
    def __init__(
            self,
            obs_dim: int,
            acs_dim: int,
            latent_dim: int,
            hidden_sizes: Tuple[int, ...],
            batch_size: int,
            lr_schedule: Callable[[float], float],
            expert_obs: np.ndarray,
            expert_acs: np.ndarray,
            is_discrete: bool,
            task: str = 'ICRL',
            regularizer_coeff: float = 0.,
            obs_select_dim: Optional[Tuple[int, ...]] = None,
            acs_select_dim: Optional[Tuple[int, ...]] = None,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            no_importance_sampling: bool = True,
            per_step_importance_sampling: bool = False,
            clip_obs: Optional[float] = 10.,
            initial_obs_mean: Optional[np.ndarray] = None,
            initial_obs_var: Optional[np.ndarray] = None,
            action_low: Optional[float] = None,
            action_high: Optional[float] = None,
            target_kl_old_new: float = -1,
            target_kl_new_old: float = -1,
            train_gail_lambda: Optional[bool] = False,
            max_seq_length: float = 10,
            eps: float = 1e-5,
            eta: float = 0.1,
            device: str = "cpu",
            log_file=None,
    ):
        super(MixtureConstraintNet, self).__init__(obs_dim=obs_dim,
                                                   acs_dim=acs_dim,
                                                   hidden_sizes=hidden_sizes,
                                                   batch_size=batch_size,
                                                   lr_schedule=lr_schedule,
                                                   expert_obs=expert_obs,
                                                   expert_acs=expert_acs,
                                                   is_discrete=is_discrete,
                                                   task=task,
                                                   regularizer_coeff=regularizer_coeff,
                                                   obs_select_dim=obs_select_dim,
                                                   acs_select_dim=acs_select_dim,
                                                   optimizer_class=optimizer_class,
                                                   optimizer_kwargs=optimizer_kwargs,
                                                   no_importance_sampling=no_importance_sampling,
                                                   per_step_importance_sampling=per_step_importance_sampling,
                                                   clip_obs=clip_obs,
                                                   initial_obs_mean=initial_obs_mean,
                                                   initial_obs_var=initial_obs_var,
                                                   action_low=action_low,
                                                   action_high=action_high,
                                                   target_kl_old_new=target_kl_old_new,
                                                   target_kl_new_old=target_kl_new_old,
                                                   train_gail_lambda=train_gail_lambda,
                                                   eps=eps,
                                                   device=device,
                                                   log_file=log_file,
                                                   build_net=False)
        self.latent_dim = latent_dim
        self.max_seq_length = max_seq_length
        self.eta = eta
        self._build()

    def _define_input_dims(self) -> None:
        self.input_obs_dim = []
        self.input_acs_dim = []
        if self.obs_select_dim is None:
            self.input_obs_dim += [i for i in range(self.obs_dim)]
        elif self.obs_select_dim[0] != -1:
            self.input_obs_dim += self.obs_select_dim
        obs_len = len(self.input_obs_dim)
        if self.acs_select_dim is None:
            self.input_acs_dim += [i for i in range(self.acs_dim)]
        elif self.acs_select_dim[0] != -1:
            self.input_acs_dim += self.acs_select_dim
        self.select_dim = self.input_obs_dim + [i + obs_len for i in self.input_acs_dim]
        self.input_dim = len(self.select_dim)
        assert self.input_dim > 0, ""

    def _build(self) -> None:

        # Create constraint function and add sigmoid at the end
        self.constraint_function = nn.Sequential(
            *create_mlp(self.input_dim + self.latent_dim, 1, list(self.hidden_sizes)),
            nn.Sigmoid()
        )
        self.constraint_function.to(self.device)

        # Create the posterior of latent code encoder. The code should be in a one-hot format, so we use
        self.rnn = nn.GRUCell(self.input_dim, self.hidden_sizes[0]).to(self.device)
        self.posterior_encoder = nn.Sequential(
            *create_mlp(self.hidden_sizes[0], self.latent_dim,
                        list(self.hidden_sizes[1:]) if len(self.hidden_sizes) > 1 else self.hidden_sizes),
            nn.Softmax()
        ).to(self.device)
        self.bce_loss = th.nn.BCELoss()

        # Build optimizer
        if self.optimizer_class is not None:
            self.optimizer = self.optimizer_class(self.parameters(), lr=self.lr_schedule(1), **self.optimizer_kwargs)
        else:
            self.optimizer = None
        if self.train_gail_lambda:
            self.criterion = nn.BCELoss()

    def forward(self, x: th.tensor) -> th.tensor:
        return self.constraint_function(x)

    def cost_function_with_code(self, obs: np.ndarray, acs: np.ndarray, codes: np.ndarray) -> np.ndarray:
        assert obs.shape[-1] == self.obs_dim, ""
        if not self.is_discrete:
            assert acs.shape[-1] == self.acs_dim, ""

        data = self.prepare_data(obs, acs)
        codes = th.tensor(codes, dtype=th.float32).to(self.device)
        function_input = th.cat([data, codes], dim=1)
        with th.no_grad():
            out = self.__call__(function_input)
        cost = 1 - out.detach().cpu().numpy()
        return cost.squeeze(axis=-1)

    def latent_function(self, obs: List[np.ndarray], acs: List[np.ndarray]) -> np.ndarray:
        """
        obs: [env_len, seq_len, obs_dim]
        acs: [env_len, seq_len, obs_dim]
        """
        assert len(obs) == len(acs)
        data_games = [self.prepare_data(obs[i], acs[i]) for i in range(len(obs))]
        seq_games = [build_rnn_input(max_seq_length=self.max_seq_length,
                                     input_data_list=data_games[i])[-1, :]
                     for i in range(len(data_games))]
        seqs = th.stack(seq_games, dim=0)  # [env_len, seq_len, obs_dim]
        rnn_batch_hidden_states = None
        with th.no_grad():
            for i in range(int(self.max_seq_length)):
                rnn_input = seqs[:, i, :]
                rnn_batch_hidden_states = self.rnn(input=rnn_input, hx=rnn_batch_hidden_states)
            latent_posterior = self.posterior_encoder(rnn_batch_hidden_states).detach().cpu().numpy()
        return latent_posterior

    def call_forward(self, x: np.ndarray):
        with th.no_grad():
            out = self.__call__(th.tensor(x, dtype=th.float32).to(self.device))
        return out

    def train_nn(
            self,
            iterations: np.ndarray,
            nominal_obs: np.ndarray,
            nominal_acs: np.ndarray,
            episode_lengths: np.ndarray,
            obs_mean: Optional[np.ndarray] = None,
            obs_var: Optional[np.ndarray] = None,
            current_progress_remaining: float = 1,
            **other_parameters,
    ) -> Dict[str, Any]:

        # Update learning rate
        self._update_learning_rate(current_progress_remaining)

        # Update normalization stats
        self.current_obs_mean = obs_mean
        self.current_obs_var = obs_var

        # Prepare data
        nominal_games = [self.prepare_data(nominal_obs[i], nominal_acs[i]) for i in range(len(nominal_obs))]
        expert_games = [self.prepare_data(self.expert_obs[i], self.expert_acs[i]) for i in range(len(self.expert_obs))]

        # build rnn input:
        nominal_seq_games = [build_rnn_input(max_seq_length=self.max_seq_length,
                                             input_data_list=nominal_games[i])
                             for i in range(len(nominal_obs))]
        nominal_seqs = th.cat(nominal_seq_games, dim=0)  # [data_num, seq_len, input_dim]
        expert_seq_games = [build_rnn_input(max_seq_length=self.max_seq_length,
                                            input_data_list=expert_games[i])
                            for i in range(len(expert_games))]
        expert_seqs = th.cat(expert_seq_games, dim=0)  # [data_num, seq_len, input_dim]

        assert 'nominal_codes' in other_parameters
        nominal_codes = th.tensor(other_parameters['nominal_codes'], dtype=th.float32). \
            reshape([-1, self.latent_dim]).to(self.device)
        assert 'expert_codes' in other_parameters
        expert_codes = th.tensor(other_parameters['expert_codes'], dtype=th.float32). \
            reshape([-1, self.latent_dim]).to(self.device)
        assert 'debug_msg' in other_parameters
        debug_msg = other_parameters['debug_msg'][0]

        # list all possible candidate codes with shape [batch_size, latent_dim, latent_dim],
        # for example, for latent_dim =2, we have [[1,0],[0,1]]*batch_size
        expert_candidate_code = build_code(code_axis=[i for i in range(self.latent_dim)],
                                           code_dim=self.latent_dim,
                                           num_envs=self.latent_dim)
        for itr in tqdm(range(iterations)):
            # # classify the expert data with posterior
            expert_code_games = []
            expert_latent_prob_games = []
            for i in range(len(expert_seq_games)):
                expert_seq_game = expert_seq_games[i]
                rnn_batch_hidden_states = None
                for i in range(int(self.max_seq_length)):  # rnn feature extractor
                    rnn_batch_input = expert_seq_game[:, i, :]
                    rnn_batch_hidden_states = self.rnn(input=rnn_batch_input, hx=rnn_batch_hidden_states)
                # the predicted prob of code id (cid) for each expert datapoint
                expert_latent_prob_game = self.posterior_encoder(rnn_batch_hidden_states).detach()
                expert_latent_prob_games.append(expert_latent_prob_game)
                # sum up the log-prob for datapoints to determine the cid of entire trajectory.
                expert_log_sum_game = th.log(expert_latent_prob_game + self.eps).sum(dim=0)
                expert_cid_game = expert_log_sum_game.argmax().repeat(len(expert_seq_game))
                # repeat the cid to label all the expert datapoints.
                expert_code_game = F.one_hot(expert_cid_game, num_classes=self.latent_dim).to(self.device)
                expert_code_games.append(expert_code_game)
            expert_codes = th.cat(expert_code_games, dim=0)
            expert_latent_prob_games = th.cat(expert_latent_prob_games, dim=0)
            tmp = expert_latent_prob_games.mean(dim=0)

            # Do a complete pass on data
            for nom_batch_indices, exp_batch_indices in self.get(nominal_seqs.shape[0], expert_seqs.shape[0]):
                # Get batch data
                nominal_data_batch = nominal_seqs[nom_batch_indices][:, -1, :]  # [data_num, input_dim]
                nominal_seqs_batch = nominal_seqs[nom_batch_indices]
                nominal_code_batch = nominal_codes[nom_batch_indices]
                expert_data_batch = expert_seqs[exp_batch_indices][:, -1, :]  # [data_num, input_dim]
                expert_seqs_batch = expert_seqs[exp_batch_indices]  # [data_num, seq_len, input_dim]
                expert_code_batch = expert_codes[exp_batch_indices]

                expert_candidate_codes = np.expand_dims(expert_candidate_code, axis=0).repeat(len(expert_seqs), axis=0)
                expert_candidate_codes = th.tensor(expert_candidate_codes, dtype=th.float32).to(self.device)
                expert_candidate_code_batch = expert_candidate_codes[exp_batch_indices]

                # Nominal predictions
                nominal_input_batch = th.cat([nominal_data_batch, nominal_code_batch], dim=1)
                nominal_preds = self.__call__(nominal_input_batch)
                nominal_loss = th.mean(th.log(nominal_preds + self.eps))

                # Latent predictions
                # rnn_batch_hidden_states = None
                # for i in range(int(self.max_seq_length)):
                #     rnn_batch_input = expert_seqs_batch[:, i, :]
                #     rnn_batch_hidden_states = self.rnn(input=rnn_batch_input, hx=rnn_batch_hidden_states)
                # expert_latent_prob = self.posterior_encoder(rnn_batch_hidden_states).detach()
                # expert_latent_pred = expert_latent_prob.argmax(dim=1)
                # expert_code_batch = F.one_hot(expert_latent_pred, num_classes=self.latent_dim).to(self.device)
                # expert_latent_others_mask = torch.ones([expert_latent_prob.shape[0], self.latent_dim]).to(
                #     self.device) - expert_latent_max_mask

                # TODO: add the lower-bound of mutual information?
                # Expert predictions
                # [batch_size, latent_dim, input_dim]
                expert_candidate_batch = expert_data_batch.unsqueeze(dim=1).repeat(1, self.latent_dim, 1)
                # [batch_size, latent_dim, input_dim+latent_dim]
                expert_input_batch = th.cat([expert_candidate_batch, expert_candidate_code_batch], dim=-1)
                expert_preds = self.__call__(expert_input_batch.reshape(
                    shape=[-1, self.latent_dim + self.input_dim])).reshape(shape=[-1, self.latent_dim])
                if 'sanity_check' in debug_msg:
                    sanity_check_expert_code_batch = 0.5 * th.ones(expert_code_batch.shape).to(self.device)
                    expert_loss_batch = -th.sum(th.log(expert_preds + self.eps) * sanity_check_expert_code_batch, dim=1)
                else:
                    expert_loss_batch = -th.sum(th.log(expert_preds + self.eps) * expert_code_batch, dim=1)
                expert_loss = expert_loss_batch.mean()
                # tmp1 = th.log(expert_preds + self.eps) * expert_latent_max_mask
                # expert_loss_1 = -th.sum(th.log(expert_preds + self.eps) * expert_latent_max_mask, dim=1)
                # tmp2 = th.log(expert_preds + self.eps) * expert_latent_others_mask
                # expert_loss_2 = 1 / (self.latent_dim - 1) * th.sum(
                #     th.log(expert_preds + self.eps) * expert_latent_others_mask, dim=1)
                # expert_loss = (expert_loss_1 + self.eta * expert_loss_2).mean()

                # add regularization
                # regularizer_loss = self.regularizer_coeff * (th.mean(1 - expert_preds) + th.mean(1 - nominal_preds))
                regularizer_loss = th.tensor(0)
                discriminator_loss = expert_loss + (1 - self.eta) * nominal_loss

                # update posterior
                rnn_batch_hidden_states = None
                for i in range(int(self.max_seq_length)):
                    rnn_batch_input = nominal_seqs_batch[:, i, :]
                    rnn_batch_hidden_states = self.rnn(input=rnn_batch_input, hx=rnn_batch_hidden_states)
                nominal_latent_prob = self.posterior_encoder(rnn_batch_hidden_states)
                posterior_loss = self.bce_loss(nominal_latent_prob, nominal_code_batch)

                # Update
                self.optimizer.zero_grad()
                discriminator_loss.backward()
                posterior_loss.backward()
                self.optimizer.step()

        bw_metrics = {"backward/cn_loss": discriminator_loss.item(),
                      "backward/posterior_loss": posterior_loss.item(),
                      "backward/expert_loss": expert_loss.item(),
                      "backward/unweighted_nominal_loss": th.mean(th.log(nominal_preds + self.eps)).item(),
                      "backward/nominal_loss": nominal_loss.item(),
                      "backward/regularizer_loss": regularizer_loss.item(),
                      "backward/nominal_preds_max": th.max(nominal_preds).item(),
                      "backward/nominal_preds_min": th.min(nominal_preds).item(),
                      "backward/nominal_preds_mean": th.mean(nominal_preds).item(),
                      "backward/expert_preds_max": th.max(expert_preds).item(),
                      "backward/expert_preds_min": th.min(expert_preds).item(),
                      "backward/expert_preds_mean": th.mean(expert_preds).item(),
                      "backward/latent_pred": th.mean(expert_latent_prob_games, dim=0).numpy().tolist()
                      }
        return bw_metrics

    def save(self, save_path):
        state_dict = dict(
            cn_network=self.constraint_function.state_dict(),
            posterior_encoder=self.posterior_encoder.state_dict(),
            rnn=self.rnn.state_dict(),
            cn_optimizer=self.optimizer.state_dict(),
            obs_dim=self.obs_dim,
            acs_dim=self.acs_dim,
            is_discrete=self.is_discrete,
            obs_select_dim=self.obs_select_dim,
            acs_select_dim=self.acs_select_dim,
            clip_obs=self.clip_obs,
            obs_mean=self.current_obs_mean,
            obs_var=self.current_obs_var,
            action_low=self.action_low,
            action_high=self.action_high,
            device=self.device,
            hidden_sizes=self.hidden_sizes,
            latent_dim=self.latent_dim,
            input_dim=self.input_dim,
        )
        th.save(state_dict, save_path)

    def _load(self, load_path):
        state_dict = th.load(load_path)
        if "cn_network" in state_dict:
            self.constraint_function.load_state_dict(dic["cn_network"])
        if "cn_optimizer" in state_dict and self.optimizer is not None:
            self.optimizer.load_state_dict(dic["cn_optimizer"])

    # Provide basic functionality to load this class
    @classmethod
    def load(
            cls,
            load_path: str,
            obs_dim: Optional[int] = None,
            acs_dim: Optional[int] = None,
            latent_dim: Optional[int] = None,
            is_discrete: bool = None,
            obs_select_dim: Optional[Tuple[int, ...]] = None,
            acs_select_dim: Optional[Tuple[int, ...]] = None,
            clip_obs: Optional[float] = None,
            obs_mean: Optional[np.ndarray] = None,
            obs_var: Optional[np.ndarray] = None,
            action_low: Optional[float] = None,
            action_high: Optional[float] = None,
            device: str = "auto"
    ):

        state_dict = th.load(load_path)
        # If value isn't specified, then get from state_dict
        if obs_dim is None:
            obs_dim = state_dict["obs_dim"]
        if acs_dim is None:
            acs_dim = state_dict["acs_dim"]
        if is_discrete is None:
            is_discrete = state_dict["is_discrete"]
        if obs_select_dim is None:
            obs_select_dim = state_dict["obs_select_dim"]
        if acs_select_dim is None:
            acs_select_dim = state_dict["acs_select_dim"]
        if clip_obs is None:
            clip_obs = state_dict["clip_obs"]
        if obs_mean is None:
            obs_mean = state_dict["obs_mean"]
        if obs_var is None:
            obs_var = state_dict["obs_var"]
        if action_low is None:
            action_low = state_dict["action_low"]
        if action_high is None:
            action_high = state_dict["action_high"]
        if device is None:
            device = state_dict["device"]
        if latent_dim is None:
            latent_dim = state_dict["latent_dim"]

        # Create network
        hidden_sizes = state_dict["hidden_sizes"]
        constraint_net = cls(
            obs_dim=obs_dim,
            acs_dim=acs_dim,
            hidden_sizes=hidden_sizes,
            batch_size=None,
            lr_schedule=None,
            expert_obs=None,
            expert_acs=None,
            optimizer_class=None,
            is_discrete=is_discrete,
            obs_select_dim=obs_select_dim,
            acs_select_dim=acs_select_dim,
            clip_obs=clip_obs,
            initial_obs_mean=obs_mean,
            initial_obs_var=obs_var,
            action_low=action_low,
            action_high=action_high,
            device=device,
            latent_dim=latent_dim
        )
        constraint_net.constraint_function.load_state_dict(state_dict["cn_network"])

        return constraint_net
