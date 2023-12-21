import random
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union, List
import numpy as np
import torch
import torch.nn.functional as F
from models.constraint_net.constraint_net import ConstraintNet
from cirl_stable_baselines3.common.torch_layers import create_mlp
from cirl_stable_baselines3.common.utils import update_learning_rate
from torch import nn
from tqdm import tqdm
# from models.nf_net.config import CONFIG as cfg
from utils.model_utils import build_code, to_np
from models.nf_net.GFlownet import FlowModel
torch.cuda.current_device()
torch.cuda._initialized = True


# from models.cflownets.loss_network import CriticTrainNetWrapper

class GFlowConstraintNet(ConstraintNet):
    def __init__(
            self,
            obs_dim: int,
            acs_dim: int,
            latent_dim: int,
            episodes: int,
            GF_hidden_sizes: int,
            hidden_sizes: Tuple[int, ...],
            batch_size: int,
            cn_lr_schedule: Callable[[float], float],
            density_lr_schedule: Callable[[float], float],
            expert_obs: np.ndarray,
            expert_acs: np.ndarray,
            is_discrete: bool,
            task: str = 'ICRL',
            regularizer_coeff: float = 0.,
            obs_select_dim: Optional[Tuple[int, ...]] = None,
            acs_select_dim: Optional[Tuple[int, ...]] = None,
            optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
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
            init_density: bool = True,
            use_expert_negative: bool = False,
            negative_weight: float = 1.0,
            sample_probing_points: bool = False,
            n_probings: int = 1,
            reverse_probing: bool = False,
            log_cost: bool = False,
            eps: float = 1e-5,
            eta: float = 0.1,
            device: str = "cpu",
            log_file=None,
            recon_obs: bool = False,
            env_configs: dict = {},
    ):
        super(GFlowConstraintNet, self).__init__(obs_dim=obs_dim,
                                                   acs_dim=acs_dim,
                                                   hidden_sizes=hidden_sizes,
                                                   batch_size=batch_size,
                                                   lr_schedule=cn_lr_schedule,
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
                                                   recon_obs=recon_obs,
                                                   build_net=False,
                                                   env_configs=env_configs, )
        self.latent_dim = latent_dim
        self.episodes = episodes
        self.GF_hidden_sizes = GF_hidden_sizes
        self.max_seq_length = max_seq_length
        self.init_density = init_density
        self.use_expert_negative = use_expert_negative
        self.negative_weight = negative_weight
        self.sample_probing_points = sample_probing_points
        self.eta = eta
        self.pivot_vectors_by_cid = {}
        self.n_probings = n_probings
        self.reverse_probing = reverse_probing
        self.log_cost = log_cost
        self.density_lr_schedule = density_lr_schedule
        self.cn_lr_schedule = cn_lr_schedule
        for aid in range(self.latent_dim):
            self.pivot_vectors_by_cid.update({aid: np.ones([self.n_probings, self.obs_dim + self.acs_dim])})
        self._build()
        self.criterion = nn.BCELoss()
        self._init_games_by_aids()
        for i in range(len(expert_obs)):
            aid = random.randrange(self.latent_dim)
            self.games_by_aids[aid].append(i)

    def _init_games_by_aids(self):
        self.games_by_aids = {}
        for aid in range(self.latent_dim):
            self.games_by_aids.update({aid: []})

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
        self._init_density_model()
        self._init_constraint_model()
        if self.train_gail_lambda:
            self.criterion = nn.BCELoss()

    def _init_constraint_model(self):
        # Create constraint function and add sigmoid at the end
        self.constraint_functions = [nn.Sequential(
            *create_mlp(self.input_dim, 1, list(self.hidden_sizes)),
            nn.Sigmoid()
        ).to(self.device) for i in range(self.latent_dim)]

        if self.optimizer_class is not None:
            self.cns_optimizers = []
            for i in range(self.latent_dim):
                self.cns_optimizers.append(self.optimizer_class(params=self.constraint_functions[i].parameters(),
                                                                lr=self.cn_lr_schedule(1),
                                                                **self.optimizer_kwargs))
        else:
            self.cns_optimizers = None

    def _init_density_model(self):
        # Creat density model
        model =  FlowModel(num_inputs=self.input_dim,
                     hidden_dim=self.GF_hidden_sizes,
                     num_cond_inputs=self.input_dim,
                     )

        if isinstance(model, nn.Linear):
            nn.init.orthogonal_(model.weight)
            if hasattr(model, 'bias') and model.bias is not None:
                model.bias.data.fill_(0)
        model.to(self.device)
        self.density_model = model

        if self.optimizer_class is not None:
            self.density_optimizer = self.optimizer_class(params=self.density_model.parameters(),
                                                          lr=self.density_lr_schedule(1),
                                                          **self.optimizer_kwargs)
        else:
            self.density_optimizer = None

    def forward(self, x: torch.tensor) -> torch.tensor:
        data = x[:, :-self.latent_dim]
        codes = x[:, -self.latent_dim:]
        outputs = []
        for i in range(self.latent_dim):
            outputs.append(self.constraint_functions[i](data).squeeze(dim=1))
        outputs = torch.stack(outputs, dim=1)
        outputs = outputs * codes
        # tmp = torch.sum(outputs, dim=1)
        return torch.sum(outputs, dim=1)

    def cost_function_with_code(self, obs: np.ndarray, acs: np.ndarray, codes: np.ndarray) -> np.ndarray:
        assert obs.shape[-1] == self.obs_dim or self.recon_obs
        if not self.is_discrete:
            assert acs.shape[-1] == self.acs_dim, ""

        data = self.prepare_data(obs, acs)
        codes = torch.tensor(codes, dtype=torch.float32).to(self.device)
        function_input = torch.cat([data, codes], dim=1)
        with torch.no_grad():
            out = self.__call__(function_input).detach().cpu().numpy()
        if self.log_cost:
            cost = -np.log(np.maximum(out, np.ones([out.shape[0]]) * 1e-8))
        else:
            cost = 1 - out
        return cost

    def latent_function(self, obs: np.ndarray, acs: np.ndarray, codes: np.ndarray):
        cids = codes.argmax(0)
        latent_signals = []
        for cid in cids:
            positive_samples = self.pivot_vectors_by_cid[cid]
            negative_samples = []
            for nid in self.pivot_vectors_by_cid.keys():
                if nid == cid:
                    continue
                negative_samples.append(self.pivot_vectors_by_cid[nid])
            negative_samples = np.stack(negative_samples, axis=0)
            latent_signals.append({'pos': positive_samples, 'neg': negative_samples})
        return latent_signals

    def call_forward(self, x: np.ndarray):
        with torch.no_grad():
            out = self.__call__(torch.tensor(x, dtype=torch.float32).to(self.device), )
        return out

    def mixture_get(self, nom_size: int, exp_size: int, neg_exp_size: int) -> np.ndarray:
        if self.batch_size is None:
            yield np.arange(nom_size), np.arange(exp_size), np.arange(neg_exp_size)
        else:
            size = min(nom_size, exp_size, neg_exp_size)
            nom_indices = np.random.permutation(nom_size)
            expert_indices = np.random.permutation(exp_size)
            neg_expert_indices = np.random.permutation(neg_exp_size)
            start_idx = 0
            while start_idx < size:
                batch_expert_indices = expert_indices[start_idx:start_idx + self.batch_size]
                batch_nom_indices = nom_indices[start_idx:start_idx + self.batch_size]
                batch_neg_expert_indices = neg_expert_indices[start_idx:start_idx + self.batch_size]
                yield batch_nom_indices, batch_expert_indices, batch_neg_expert_indices
                start_idx += self.batch_size

    def train_traj_nn(
            self,
            env,
            iterations: np.ndarray,
            nominal_obs: np.ndarray,
            nominal_acs: np.ndarray,
            episode_lengths: np.ndarray,
            obs_mean: Optional[np.ndarray] = None,
            obs_var: Optional[np.ndarray] = None,
            current_progress_remaining: float = 1,
            **other_parameters,
    ) -> Dict[str, Any]:
        bw_metrics = {}
        # Update learning rate
        self._update_learning_rate(current_progress_remaining)

        if self.init_density:
            # init the density model at each iteration
            self._init_density_model()

        # Update normalization stats
        self.current_obs_mean = obs_mean
        self.current_obs_var = obs_var

        # debug msg for sanity check
        debug_msg = other_parameters['debug_msg'][0]

        # Prepare data
        nominal_data_games = [self.prepare_data(nominal_obs[i], nominal_acs[i]) for i in range(len(nominal_obs))]
        expert_data_games = [self.prepare_data(self.expert_obs[i], self.expert_acs[i]) for i in
                             range(len(self.expert_obs))]

        nominal_data = torch.cat(nominal_data_games, dim=0)
        expert_data = torch.cat(expert_data_games, dim=0)

        assert 'nominal_codes' in other_parameters
        nominal_code_games = [torch.tensor(other_parameters['nominal_codes'][i]).to(self.device) for i in
                              range(len(other_parameters['nominal_codes']))]

        # save training data for different codes
        nominal_data_by_cids = {}
        for aid in range(self.latent_dim):
            nominal_data_by_cids.update({aid: []})
        nominal_log_prob_by_aid = {}
        for aid in range(self.latent_dim):
            nominal_log_prob_by_aid.update({aid: None})
        expert_data_by_cids = {}
        for aid in range(self.latent_dim):
            expert_data_by_cids.update({aid: []})

        for i in range(len(nominal_data_games)):
            nominal_data_game = nominal_data_games[i]
            nominal_code_game = nominal_code_games[i]
            nominal_cid = nominal_code_game[0].argmax()
            nominal_data_by_cids[nominal_cid.item()].append(nominal_data_game)

        density_loss = 0
        density_loss_metric = 0

        for episode in tqdm(range(self.episodes), ncols=100):
            for choice in range(0,self.latent_dim):
                for game in nominal_data_by_cids[choice]:
                    # Predict P_F, P_B
                    P_F_s = self.density_model(torch.zeros(nominal_data_game[0].size()).to(self.device))
                    total_P_F = 0
                    total_P_B = 0
                    temp = 0

                    reward_list = []
                    for i in range(0,len(game)):
                        temp+=(1/len(game))
                        reward_list.append(temp)

                    for i in range(0,len(game)-1):
                        P_F_s = self.density_model(game[i])
                        # Accumulate the P_F sum
                        total_P_F = total_P_F + P_F_s

                        # 使用lstm记录历史信息
                        # temp.append(game[i])
                        # output_dim=torch.zeros(game[0].size()).shape[0]
                        # input_dim=torch.cat(temp, dim=0).shape[0]
                        # lstm = torch.nn.LSTM(input_dim,output_dim,batch_first=False, num_layers=1).to(self.device)
                        # output,(hidden,c)=lstm(torch.unsqueeze(torch.unsqueeze(torch.cat(temp, dim=0),dim=0),dim=0).to(self.device))

                        # Go to the next state
                        new_state = game[i+1]
                        reward = reward_list[i]
                        # if reward == 1:
                        #     break

                        # We recompute P_F and P_B for new_game
                        P_B_s = self.density_model(new_state)

                        # Here we accumulate P_B, going backwards from `new_state`. We're also just
                        # going to use opposite semantics for the backward policy.
                        total_P_B = total_P_B + P_B_s

                        # We're done with the trajectory, let's compute its loss. Since the reward can
                        #  sometimes be zero, instead of log(0) we'll clip the log-reward to -20.
                        loss = torch.mean((total_P_F - torch.log(torch.tensor(reward).float()).clip(-20) - total_P_B).pow(2))
                        density_loss = density_loss + loss

                    self.density_optimizer.zero_grad()
                    density_loss.backward()
                    print(density_loss)
                    self.density_optimizer.step()
                    density_loss_metric = density_loss
                    density_loss = 0

        bw_metrics.update({"backward/density_loss": density_loss_metric})

        # for episode in tqdm(range(self.episodes), ncols=100):
        #     for choice in range(0,self.latent_dim):
        #         for game in nominal_data_by_cids[choice]:
        #             action_dim = torch.zeros(nominal_acs[0].size())
        #             state_dim = torch.zeros(nominal_obs[0].size())
        #             max_action = float(env.action_space.high[0])
        #
        #             # --------  define policy --------
        #             hidden_dim = cfg['hidden_dim']
        #             uniform_action_size = cfg['uniform_action_size']
        #             CFlownet = CFN(state_dim, action_dim, hidden_dim, max_action, uniform_action_size, self.density_model)
        #             density_model_train_net = CriticTrainNetWrapper(CFlownet.density_model)
        #             reward_list = []
        #             not_done = []
        #             done_true = []
        #             for i in range(0,len(game)-1):
        #                 if i == (len(game)-2):
        #                     reward_list.append(1)
        #                     not_done.append(0)
        #                     done_true.append(1)
        #                 else:
        #                     reward_list.append(0)
        #                     not_done.append(1)
        #                     done_true.append(0)
        #             for i in range(0, len(game)-1):
        #                 density_loss += CFlownet.train(nominal_obs[i],nominal_acs[i],reward_list[i],nominal_obs[i+1],density_model_train_net,not_done[i],done_true[i])
        #
        #         self.density_optimizer.zero_grad()
        #         density_loss.backward()
        #         self.density_optimizer.step()
        #         density_loss_metric = density_loss
        #         density_loss = 0
        #
        # bw_metrics.update({"backward/density_loss": density_loss_metric})

        # scan through the expert data and classify them
        # list all possible candidate codes with shape [batch_size, latent_dim, latent_dim],
        # for example, for latent_dim =2, we have [[1,0],[0,1]]*batch_size


        expert_candidate_code = build_code(code_axis=[i for i in range(self.latent_dim)],
                                           code_dim=self.latent_dim,
                                           num_envs=self.latent_dim)
        expert_candidate_code = torch.tensor(expert_candidate_code).to(self.device)
        expert_sum_log_prob_by_games = []
        for i in range(len(expert_data_games)):
            expert_data_game_repeat = expert_data_games[i].unsqueeze(dim=1).repeat(1, self.latent_dim, 1)
            expert_code_game = expert_candidate_code.unsqueeze(dim=0).repeat(len(expert_data_games[i]), 1, 1)
            expert_log_prob = self.density_model.log_probs(
                inputs=expert_data_game_repeat.reshape([-1, self.input_dim]),
                cond_inputs=expert_code_game.reshape(
                    [-1, self.latent_dim]))
            # sum up the log-prob for datapoints to determine the aid of entire trajectory.
            expert_log_sum_game = expert_log_prob.reshape([-1, self.latent_dim, 1]).sum(dim=0).squeeze(dim=-1)
            expert_sum_log_prob_by_games.append(to_np(expert_log_sum_game))
        expert_sum_log_prob_by_games = np.asarray(expert_sum_log_prob_by_games)
        expert_code_games = [None for i in range(len(expert_data_games))]
        self._init_games_by_aids()
        for expert_aid in range(self.latent_dim):
            if 'sanity_check' in debug_msg or 'semi_check' in debug_msg:
                top_ids = []
                for i in range(len(expert_data_games)):
                    ground_truth_id = np.argmax(other_parameters['expert_codes'][i][0])
                    if ground_truth_id == expert_aid:
                        top_ids.append(i)
            elif 'robust_check' in debug_msg:
                robust_weight = float(debug_msg.split('_')[2])
                top_ids = []
                for i in range(len(expert_data_games)):
                    ground_truth_id = np.argmax(other_parameters['expert_codes'][i][0])
                    if ground_truth_id == expert_aid:
                        if random.random() > robust_weight:  # add the right ids for 90% of times
                            top_ids.append(i)
                    else:
                        if random.random() < robust_weight:  # add the wrong ids for 10% of times
                            top_ids.append(i)
            else:
                other_dims = list(range(self.latent_dim))
                other_dims.remove(expert_aid)
                diff = np.zeros([len(expert_data_games)])
                for k in other_dims:
                    diff += expert_sum_log_prob_by_games[:, expert_aid] - expert_sum_log_prob_by_games[:, k]

                top_ids = np.argsort(diff)[::-1][:int(len(expert_data_games) / self.latent_dim)]
            for i in top_ids:
                print("expert game: {0}, cid: {1}, log_sum: {2}".format(i,
                                                                        expert_aid,
                                                                        expert_sum_log_prob_by_games[i]),
                      file=self.log_file, flush=True)
                self.games_by_aids[expert_aid].append(i)
                # if expert_data_by_cid[expert_aid] is None:
                #     expert_data_by_cid[expert_aid] = expert_data_games[i]
                # else:
                #     expert_data_by_cid[expert_aid] = torch.cat([expert_data_by_cid[expert_aid],
                #                                                 expert_data_games[i]], dim=0)
                expert_data_by_cids[expert_aid].append(expert_data_games[i])
                expert_cid_game = torch.tensor(np.repeat(expert_aid, len(expert_data_games[i])))
                # repeat the cid to label all the expert datapoints.
                expert_code_game = F.one_hot(expert_cid_game.to(torch.int64), num_classes=self.latent_dim).to(
                    self.device)
                expert_code_games[i] = expert_code_game

        nominal_data_by_cids = {}
        for aid in range(self.latent_dim):
            nominal_data_by_cids.update({aid: []})
        nominal_log_prob_by_aid = {}
        for aid in range(self.latent_dim):
            nominal_log_prob_by_aid.update({aid: None})

        for i in range(len(nominal_data_games)):
            nominal_data_game = nominal_data_games[i]
            nominal_code_game = nominal_code_games[i]
            nominal_cid = nominal_code_game[0].argmax()
            nominal_data_by_cids[nominal_cid.item()].append(nominal_data_game)
            # if nominal_data_by_cids[nominal_cid.item()] is None:
            #     nominal_data_by_cids[nominal_cid.item()] = nominal_data_game
            # else:
            #     nominal_data_by_cids[nominal_cid.item()] = torch.cat([nominal_data_by_cids[nominal_cid.item()],
            #                                                           nominal_data_game], dim=0)
        # get some probing points
        if self.sample_probing_points:  # sample the probing points
            nominal_log_prob_by_aid = None
            print('Sampling probing points.', flush=True, file=self.log_file)
        else:  # scan through the nominal data and pick some probing points
            print('Predicting probing points.', flush=True, file=self.log_file)
            for nominal_cid in nominal_data_by_cids.keys():
                nominal_data_cid = torch.cat(nominal_data_by_cids[nominal_cid], dim=0)
                nominal_code_cid = F.one_hot(torch.tensor([nominal_cid] * nominal_data_cid.shape[0]),
                                             num_classes=self.latent_dim).to(torch.float32).to(self.device)
                log_prob_game = self.density_model.log_probs(inputs=nominal_data_cid,
                                                            cond_inputs=nominal_code_cid)
                nominal_reverse_cid_game = torch.ones(size=nominal_code_cid.shape).to(self.device) - nominal_code_cid
                reverse_log_prob_game = self.density_model.log_probs(inputs=nominal_data_cid,
                                                                    cond_inputs=nominal_reverse_cid_game)
                if nominal_log_prob_by_aid[nominal_cid] is None:
                    if self.reverse_probing:
                        nominal_log_prob_by_aid[nominal_cid] = -reverse_log_prob_game
                    else:
                        nominal_log_prob_by_aid[nominal_cid] = log_prob_game - reverse_log_prob_game
                else:
                    if self.reverse_probing:
                        nominal_log_prob_by_aid[nominal_cid] = torch.cat(
                            [nominal_log_prob_by_aid[nominal_cid], -reverse_log_prob_game], dim=0)
                    else:
                        nominal_log_prob_by_aid[nominal_cid] = torch.cat(
                            [nominal_log_prob_by_aid[nominal_cid], log_prob_game - reverse_log_prob_game], dim=0)
        for aid in range(self.latent_dim):
            if self.sample_probing_points:
                cond_inputs = F.one_hot(torch.tensor([aid] * self.n_probings), num_classes=self.latent_dim).to(
                    torch.float32).to(self.device)
                pivot_points = self.density_model.sample(num_samples=self.n_probings, noise=None,
                                                         cond_inputs=cond_inputs)
            else:
                reverse_log_prob_cid = nominal_log_prob_by_aid[aid]
                _, topk = reverse_log_prob_cid.squeeze(dim=-1).topk(self.n_probings, dim=0, largest=True, sorted=False)
                pivot_points = torch.cat(nominal_data_by_cids[aid], dim=0)[topk]
            self.pivot_vectors_by_cid.update({aid: to_np(pivot_points)})
            print('aid: {0}, pivot_vectors is {1}'.format(aid, self.pivot_vectors_by_cid[aid].mean(axis=0)),
                  flush=True, file=self.log_file)
            if expert_data_by_cids[aid] is None:
                continue
            for itr in tqdm(range(iterations)):
                discriminator_loss_record, expert_loss_record, nominal_loss_record, \
                regularizer_loss_record, nominal_preds_record, expert_preds_record = [], [], [], [], [], []

                for gid in range(min(len(nominal_data_by_cids[aid]), len(expert_data_by_cids[aid]))):
                    nominal_data_cid_game = nominal_data_by_cids[aid][gid]
                    expert_data_cid_game = expert_data_by_cids[aid][gid]

                    # Do a complete pass on data
                    if self.use_expert_negative:  # if we treat the expert data of other cid as nominal data
                        other_cids = [i for i in range(self.latent_dim)]
                        other_cids.remove(aid)
                        expert_data_for_other_aids = torch.cat([expert_data_by_cids[od][gid] for od in other_cids],
                                                               dim=0)
                        expert_data_for_other_aids_size = expert_data_for_other_aids.shape[0]
                    else:
                        expert_data_for_other_aids = None
                        expert_data_for_other_aids_size = max(nominal_data_cid_game.shape[0],
                                                              expert_data_cid_game.shape[0])
                    nominal_preds_game = []
                    expert_preds_game = []
                    for nom_batch_indices, exp_batch_indices, neg_exp_batch_indices in self.mixture_get(
                            nominal_data_cid_game.shape[0],
                            expert_data_cid_game.shape[0],
                            expert_data_for_other_aids_size):
                        # Get batch data
                        nominal_data_batch = nominal_data_cid_game[nom_batch_indices]
                        expert_data_batch = expert_data_cid_game[exp_batch_indices]
                        # Make predictions
                        nom_cid_code = build_code(code_axis=[aid for i in range(len(nom_batch_indices))],
                                                  code_dim=self.latent_dim,
                                                  num_envs=len(nom_batch_indices))
                        nom_cid_code = torch.tensor(nom_cid_code).to(self.device)
                        nominal_preds = self.__call__(torch.cat([nominal_data_batch, nom_cid_code], dim=1))
                        nominal_preds_record.append(to_np(nominal_preds))
                        nominal_preds_game.append(nominal_preds)
                        expert_cid_code = build_code(code_axis=[aid for i in range(len(exp_batch_indices))],
                                                     code_dim=self.latent_dim,
                                                     num_envs=len(exp_batch_indices))
                        expert_cid_code = torch.tensor(expert_cid_code).to(self.device)
                        expert_preds = self.__call__(torch.cat([expert_data_batch, expert_cid_code], dim=1))
                        expert_preds_record.append(to_np(expert_preds))
                        expert_preds_game.append(expert_preds)

                    # Calculate loss
                    nominal_preds_game = torch.cat(nominal_preds_game, dim=0)
                    expert_preds_game = torch.cat(expert_preds_game, dim=0)
                    nominal_loss = self.criterion(nominal_preds_game,
                                                  torch.zeros(*nominal_preds_game.size()).to(self.device))
                    nominal_loss_record.append(nominal_loss.item())
                    expert_loss = self.criterion(expert_preds_game,
                                                 torch.ones(*expert_preds_game.size()).to(self.device))
                    expert_loss_record.append(expert_loss.item())
                    regularizer_loss = self.regularizer_coeff * (
                            torch.mean(1 - expert_preds_game) + torch.mean(1 - nominal_preds_game))
                    regularizer_loss_record.append(regularizer_loss.item())

                    if self.use_expert_negative:
                        neg_expert_data_batch = expert_data_for_other_aids[neg_exp_batch_indices]
                        neg_expert_cid_code = build_code(code_axis=[aid for i in range(len(neg_exp_batch_indices))],
                                                         code_dim=self.latent_dim,
                                                         num_envs=len(neg_exp_batch_indices))
                        neg_expert_cid_code = torch.tensor(neg_expert_cid_code).to(self.device)
                        neg_expert_preds = self.__call__(torch.cat([neg_expert_data_batch, neg_expert_cid_code], dim=1))
                        neg_exp_loss = self.criterion(neg_expert_preds,
                                                      torch.zeros(*neg_expert_preds.size()).to(self.device))
                        discriminator_loss = (expert_loss + self.negative_weight * nominal_loss +
                                              self.negative_weight * neg_exp_loss) + regularizer_loss
                    else:
                        discriminator_loss = (expert_loss + nominal_loss) + regularizer_loss
                    discriminator_loss_record.append(discriminator_loss.item())

                    # Update
                    for model_num in range(len(self.constraint_functions)):
                        self.constraint_functions[model_num].zero_grad()
                        self.cns_optimizers[model_num].zero_grad()
                    discriminator_loss.backward()
                    for model_num in range(len(self.constraint_functions)):
                        self.cns_optimizers[model_num].step()

            parameters_info = []
            for model_num in range(len(self.constraint_functions)):
                constraint_function = self.constraint_functions[model_num]
                for k, v in constraint_function.named_parameters():
                    if v.grad is not None:
                        parameters_info.append("cid-{0}_{1}:{2}".format(model_num, k, torch.mean(v.grad)))
                    else:
                        parameters_info.append("cid-{0}-{1}:{2}".format(model_num, k, v.grad))

            print(parameters_info, file=self.log_file, flush=True)
            discriminator_loss_record = np.asarray(discriminator_loss_record)
            expert_loss_record = np.asarray(expert_loss_record)
            nominal_loss_record = np.asarray(discriminator_loss_record)
            regularizer_loss_record = np.asarray(regularizer_loss_record)
            # expert_preds_record = np.asarray(expert_preds_record)
            # nominal_preds_record = np.asarray(nominal_preds_record)
            expert_preds_record = np.concatenate(expert_preds_record, axis=0)
            nominal_preds_record = np.concatenate(nominal_preds_record, axis=0)
            bw_metrics.update({"backward/cid:{0}/cn_loss".format(aid): discriminator_loss_record.mean(),
                               "backward/cid:{0}/e_loss".format(aid): expert_loss_record.mean(),
                               "backward/cid:{0}/n_loss".format(aid): nominal_loss_record.mean(),
                               "backward/cid:{0}/r_loss".format(aid): regularizer_loss_record.mean(),
                               "backward/cid:{0}/n_pred_max".format(aid): np.max(nominal_preds_record).item(),
                               "backward/cid:{0}/n_pred_min".format(aid): np.min(nominal_preds_record).item(),
                               "backward/cid:{0}/n_pred_mean".format(aid): np.mean(nominal_preds_record).item(),
                               "backward/cid:{0}/e_pred_max".format(aid): np.max(expert_preds_record).item(),
                               "backward/cid:{0}/e_pred_min".format(aid): np.min(expert_preds_record).item(),
                               "backward/cid:{0}/e_pred_mean".format(aid): np.mean(expert_preds_record).item(),
                               })
        return bw_metrics

    def save(self, save_path):
        state_dict = dict(
            cn_network=[self.constraint_functions[i].state_dict() for i in range(self.latent_dim)],
            density_model=self.density_model.state_dict(),
            cn_optimizers=[self.cns_optimizers[i].state_dict() for i in range(self.latent_dim)],
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
            GF_hidden_sizes=self.GF_hidden_sizes,
        )
        torch.save(state_dict, save_path)

    def _update_learning_rate(self, current_progress_remaining) -> None:
        self.current_progress_remaining = current_progress_remaining
        for i in range(self.latent_dim):
            update_learning_rate(self.cns_optimizers[i], self.cn_lr_schedule(current_progress_remaining))
        update_learning_rate(self.density_optimizer, self.density_lr_schedule(current_progress_remaining))
        print("The updated learning rate is density: {0}/ CN: {0}.".format(
            self.density_lr_schedule(current_progress_remaining),
            self.cn_lr_schedule(current_progress_remaining),
        ), file=self.log_file, flush=True)

    def _load(self, load_path):
        state_dict = torch.load(load_path)
        if "cn_network" in state_dict:
            self.constraint_functions.load_state_dict(dic["cn_network"])
        if "cn_optimizer" in state_dict and self.cns_optimizer is not None:
            self.cns_optimizer.load_state_dict(dic["cn_optimizer"])

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

        state_dict = torch.load(load_path)
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
        constraint_net.constraint_functions.load_state_dict(state_dict["cn_network"])

        return constraint_net
