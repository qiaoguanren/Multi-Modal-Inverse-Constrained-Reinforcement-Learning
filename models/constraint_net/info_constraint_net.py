import os
import random
from itertools import accumulate
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union
import numpy as np
import torch
import torch as th
import torch.nn.functional as F
from models.constraint_net.constraint_net import ConstraintNet
from cirl_stable_baselines3.common.torch_layers import create_mlp
from cirl_stable_baselines3.common.utils import update_learning_rate
from torch import nn
from tqdm import tqdm


class InfoConstraintNet(ConstraintNet):
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
            eps: float = 1e-5,
            device: str = "cpu",
            log_file=None,
    ):
        super(InfoConstraintNet, self).__init__(obs_dim=obs_dim,
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
        self.input_dims = len(self.select_dim)
        assert self.input_dims > 0, ""

    def _build(self) -> None:

        # Create constraint function and add sigmoid at the end
        self.constraint_function = nn.Sequential(
            *create_mlp(self.input_dims + self.latent_dim, 1, self.hidden_sizes),
            nn.Sigmoid()
        )
        self.constraint_function.to(self.device)

        # Create the posterior of latent code encoder. The code should be in a one-hot format, so we use softmax
        self.posterior_encoder = nn.Sequential(
            *create_mlp(self.input_dims, self.latent_dim, self.hidden_sizes),
            nn.Softmax()
        ).to(self.device)
        self.bce_loss = torch.nn.BCELoss()

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
        function_input = torch.cat([data, codes], dim=1)
        with th.no_grad():
            out = self.__call__(function_input)
        cost = 1 - out.detach().cpu().numpy()
        return cost.squeeze(axis=-1)

    def latent_function(self, obs: np.ndarray, acs: np.ndarray) -> np.ndarray:
        assert obs.shape[-1] == self.obs_dim, ""
        if not self.is_discrete:
            assert acs.shape[-1] == self.acs_dim, ""

        data = self.prepare_data(obs, acs)
        with th.no_grad():
            out = self.posterior_encoder(data)
        code_posterior = out.detach().cpu().numpy()
        return code_posterior

    def call_forward(self, x: np.ndarray):
        with th.no_grad():
            out = self.__call__(th.tensor(x, dtype=th.float32).to(self.device))
        return out

    def train_traj_nn(
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
        nominal_data = self.prepare_data(nominal_obs, nominal_acs)
        expert_data = self.prepare_data(self.expert_obs, self.expert_acs)
        assert 'nominal_codes' in other_parameters
        nominal_codes = th.tensor(other_parameters['nominal_codes'], dtype=th.float32).to(self.device)
        # assert 'expert_codes' in other_parameters
        # expert_codes = th.tensor(other_parameters['expert_codes'], dtype=th.float32).to(self.device)
        # expert_codes = th.tensor(np.eye(self.latent_dim)[np.random.choice(self.latent_dim, expert_data.shape[0])],
        #                          dtype=th.float32).to(self.device)

        # Save current network predictions if using importance sampling
        if self.importance_sampling:
            with th.no_grad():
                nominal_input = torch.cat([nominal_data, nominal_codes], dim=1)
                start_preds = self.forward(nominal_input).detach()

        early_stop_itr = iterations
        discriminator_loss = th.tensor(np.inf)
        for itr in tqdm(range(iterations)):
            # Compute IS weights
            assert self.importance_sampling is False
            if self.importance_sampling:
                with th.no_grad():
                    nominal_input = torch.cat([nominal_data, nominal_codes], dim=1)
                    current_preds = self.forward(nominal_input).detach()
                is_weights, kl_old_new, kl_new_old = self.compute_is_weights(start_preds.clone(),
                                                                             current_preds.clone(),
                                                                             episode_lengths)
                # Break if kl is very large
                if ((self.target_kl_old_new != -1 and kl_old_new > self.target_kl_old_new) or
                        (self.target_kl_new_old != -1 and kl_new_old > self.target_kl_new_old)):
                    early_stop_itr = itr
                    break
            else:
                is_weights = th.ones(nominal_data.shape[0]).to(self.device)

            # Do a complete pass on data
            for nom_batch_indices, exp_batch_indices in self.get(nominal_data.shape[0], expert_data.shape[0]):
                # Get batch data
                nominal_data_batch = nominal_data[nom_batch_indices]
                nominal_code_batch = nominal_codes[nom_batch_indices]
                expert_data_batch = expert_data[exp_batch_indices]
                expert_latent_prob = self.posterior_encoder(expert_data_batch).detach()
                expert_latent_pred = expert_latent_prob.argmax(dim=1)
                expert_code_batch = F.one_hot(expert_latent_pred, num_classes=self.latent_dim).to(self.device)
                is_batch = is_weights[nom_batch_indices][..., None]

                # Make predictions
                nominal_input_batch = torch.cat([nominal_data_batch, nominal_code_batch], dim=1)
                nominal_preds = self.__call__(nominal_input_batch)
                expert_input_batch = torch.cat([expert_data_batch, expert_code_batch], dim=1)
                expert_preds = self.__call__(expert_input_batch)

                expert_loss = -th.mean(th.log(expert_preds + self.eps))
                nominal_loss = th.mean(is_batch * th.log(nominal_preds + self.eps))
                # regularizer_loss = self.regularizer_coeff * (th.mean(1 - expert_preds) + th.mean(1 - nominal_preds))
                regularizer_loss = th.tensor(0)
                discriminator_loss = expert_loss + nominal_loss + regularizer_loss

                # update posterior
                posterior_output = self.posterior_encoder(nominal_data_batch)
                posterior_loss = self.bce_loss(posterior_output, nominal_code_batch)

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
                      "backward/is_mean": th.mean(is_weights).detach().item(),
                      "backward/is_max": th.max(is_weights).detach().item(),
                      "backward/is_min": th.min(is_weights).detach().item(),
                      "backward/nominal_preds_max": th.max(nominal_preds).item(),
                      "backward/nominal_preds_min": th.min(nominal_preds).item(),
                      "backward/nominal_preds_mean": th.mean(nominal_preds).item(),
                      "backward/expert_preds_max": th.max(expert_preds).item(),
                      "backward/expert_preds_min": th.min(expert_preds).item(),
                      "backward/expert_preds_mean": th.mean(expert_preds).item(), }
        if self.importance_sampling:
            stop_metrics = {"backward/kl_old_new": kl_old_new.item(),
                            "backward/kl_new_old": kl_new_old.item(),
                            "backward/early_stop_itr": early_stop_itr}
            bw_metrics.update(stop_metrics)

        return bw_metrics

    def save(self, save_path):
        state_dict = dict(
            cn_network=self.constraint_function.state_dict(),
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
            hidden_sizes=self.hidden_sizes
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
            device=device
        )
        constraint_net.constraint_function.load_state_dict(state_dict["cn_network"])

        return constraint_net
