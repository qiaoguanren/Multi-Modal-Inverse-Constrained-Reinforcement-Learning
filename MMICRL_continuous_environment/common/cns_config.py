import gym
import numpy as np
from utils.data_utils import get_input_features_dim
from utils.env_utils import is_commonroad, get_obs_feature_names


def get_cns_config(config, train_env, expert_obs, expert_acs, env_configs, log_file):
    # Initialize constraint net, true constraint net
    all_obs_feature_names = get_obs_feature_names(train_env, config['env']['train_env_id'])
    print("The observed features are: {0}".format(all_obs_feature_names), file=log_file, flush=True)

    if config['running']['store_by_game']:
        expert_obs_mean = np.mean(np.concatenate(expert_obs, axis=0), axis=0).tolist()
    else:
        expert_obs_mean = np.mean(expert_obs, axis=0).tolist()
    expert_obs_mean = ['%.5f' % elem for elem in expert_obs_mean]
    if len(all_obs_feature_names) == len(expert_obs_mean):
        expert_obs_name_mean = dict(zip(all_obs_feature_names, expert_obs_mean))
    else:
        expert_obs_name_mean = expert_obs_mean
    print("The expert features means are: {0}".format(expert_obs_name_mean),
          file=log_file,
          flush=True)

    recon_obs = config['CN']['recon_obs'] if 'recon_obs' in config['CN'].keys() else False
    is_discrete = isinstance(train_env.action_space, gym.spaces.Discrete)
    print('is_discrete', is_discrete, file=log_file, flush=True)
    if recon_obs:
        obs_dim = env_configs['map_height'] * env_configs['map_width']
    else:
        obs_dim = train_env.observation_space.shape[0]
    acs_dim = train_env.action_space.n if is_discrete else train_env.action_space.shape[0]
    action_low, action_high = None, None
    if isinstance(train_env.action_space, gym.spaces.Box):
        action_low, action_high = train_env.action_space.low, train_env.action_space.high
    cn_lr_schedule = lambda x: (config['CN']['anneal_clr_by_factor'] ** (config['running']['n_iters'] * (1 - x))) \
                               * config['CN']['cn_learning_rate']
    density_lr_schedule = lambda x: (config['CN']['anneal_clr_by_factor'] ** (config['running']['n_iters'] * (1 - x))) \
                               * config['CN']['density_learning_rate']

    cn_obs_select_name = config['CN']['cn_obs_select_name']
    print("Selecting obs features are : {0}".format(cn_obs_select_name if cn_obs_select_name is not None else 'all'),
          file=log_file, flush=True)
    cn_obs_select_dim = get_input_features_dim(feature_select_names=cn_obs_select_name,
                                               all_feature_names=all_obs_feature_names)
    cn_acs_select_name = config['CN']['cn_acs_select_name']
    print("Selecting acs features are : {0}".format(cn_acs_select_name if cn_acs_select_name is not None else 'all'),
          file=log_file, flush=True)
    cn_acs_select_dim = get_input_features_dim(feature_select_names=cn_acs_select_name,
                                               all_feature_names=['a_ego_0', 'a_ego_1'] if is_commonroad(
                                                   env_id=config['env']['train_env_id']) else None)

    if config['group'] == "MEICRL" or config['group'] == "GFICRL" or config['group'] == "ICRL" or config['group'] == "Binary" or config['group'] == "InfoICRL":
        cn_parameters = {
            'obs_dim': obs_dim,
            'acs_dim': acs_dim,
            'hidden_sizes': config['CN']['cn_layers'],
            'batch_size': config['CN']['cn_batch_size'],
            'cn_lr_schedule': cn_lr_schedule,
            'density_lr_schedule': density_lr_schedule,
            'expert_obs': expert_obs,  # select obs at a time step t
            'expert_acs': expert_acs,  # select acs at a time step t
            'is_discrete': is_discrete,
            'regularizer_coeff': config['CN']['cn_reg_coeff'],
            'obs_select_dim': cn_obs_select_dim,
            'acs_select_dim': cn_acs_select_dim,
            'clip_obs': config['CN']['clip_obs'],
            'initial_obs_mean': None if not config['CN']['cn_normalize'] else np.zeros(obs_dim),
            'initial_obs_var': None if not config['CN']['cn_normalize'] else np.ones(obs_dim),
            'action_low': action_low,
            'action_high': action_high,
            'target_kl_old_new': config['CN']['cn_target_kl_old_new'],
            'target_kl_new_old': config['CN']['cn_target_kl_new_old'],
            'train_gail_lambda': config['CN']['train_gail_lambda'],
            'eps': config['CN']['cn_eps'],
            'device': config['device'],
            'task': config['task'],
            'log_file': log_file,
            'recon_obs': config['CN']['recon_obs'],
            'env_configs': env_configs,
        }

    return cn_parameters