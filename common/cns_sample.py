import numpy as np
from cirl_stable_baselines3.common import vec_env
from cirl_stable_baselines3.common.vec_env import VecEnv
from utils.model_utils import build_code


def sample_from_multi_agents(agents, latent_dim, env, rollouts, deterministic=False, store_by_game=False,
                             **sample_parameters):
    if isinstance(env, vec_env.VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    all_orig_obs, all_obs, all_acs, all_rs, all_codes = [], [], [], [], []
    sum_rewards, lengths = [], []
    for i in range(rollouts):
        cid = i % latent_dim
        code = build_code(code_axis=[cid],
                          code_dim=latent_dim,
                          num_envs=1)
        agent = agents[cid]
        done, states = False, None
        if not isinstance(env, VecEnv) or i == 0:  # Avoid double reset, as VecEnv are reset automatically
            obs = env.reset()
        episode_sum_reward = 0.0
        episode_length = 0
        origin_obs_game, obs_game, acs_game, rs_game, codes_game = [], [], [], [], []
        while not done:
            pred_input = np.concatenate([obs, code], axis=1)
            action, states = agent.predict(pred_input, state=states, deterministic=deterministic)
            origin_obs_game.append(env.get_original_obs())
            obs_game.append(obs)
            acs_game.append(action)
            codes_game.append(code)
            # action_code = np.concatenate([actions, codes], axis=1)
            obs, reward, done, _info = env.step_with_code(action, code)
            if 'admissible_actions' in _info[0].keys():
                agent.admissible_actions = _info[0]['admissible_actions']
            rs_game.append(reward)
            episode_sum_reward += reward
            episode_length += 1

        origin_obs_game = np.squeeze(np.array(origin_obs_game), axis=1)
        obs_game = np.squeeze(np.array(obs_game), axis=1)
        acs_game = np.squeeze(np.array(acs_game), axis=1)
        rs_game = np.squeeze(np.asarray(rs_game))
        all_orig_obs.append(origin_obs_game)
        all_obs.append(obs_game)
        all_acs.append(acs_game)
        all_rs.append(rs_game)
        codes_game = np.squeeze(np.asarray(codes_game))
        all_codes.append(codes_game)

        sum_rewards.append(episode_sum_reward)
        lengths.append(episode_length)

    return all_orig_obs, all_obs, all_acs, all_rs, all_codes, sum_rewards, lengths


def sample_from_agent(agent, env, rollouts, deterministic=False, store_by_game=False, **sample_parameters):
    if isinstance(env, vec_env.VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    all_orig_obs, all_obs, all_acs, all_rs = [], [], [], []
    if sample_parameters['store_code']:
        all_codes = []

    sum_rewards, lengths = [], []
    for i in range(rollouts):
        # Avoid double reset, as VecEnv are reset automatically
        if i == 0:
            obs = env.reset()
            if sample_parameters['store_code']:
                code = sample_parameters['init_codes']

        done, states = False, None
        episode_sum_reward = 0.0
        episode_length = 0
        if store_by_game:
            origin_obs_game = []
            obs_game = []
            acs_game = []
            rs_game = []
            if sample_parameters['store_code']:
                codes_game = []
        while not done:
            if sample_parameters['store_code']:
                pred_input = np.concatenate([obs, code], axis=1)
            else:
                pred_input = obs
            action, states = agent.predict(pred_input, state=states, deterministic=deterministic)
            if store_by_game:
                origin_obs_game.append(env.get_original_obs())
                obs_game.append(obs)
                acs_game.append(action)
                if sample_parameters['store_code']:
                    codes_game.append(code)
            else:
                all_orig_obs.append(env.get_original_obs())
                all_obs.append(obs)
                all_acs.append(action)
                if sample_parameters['store_code']:
                    all_codes.append(code)
            # action_code = np.concatenate([actions, codes], axis=1)
            if sample_parameters['store_code']:
                obs, reward, done, _info = env.step(action, code)
                code = np.asarray([_info[0]["new_code"]])
            else:
                obs, reward, done, _info = env.step(action)
            agent.admissible_actions = _info[0]['admissible_actions']
            if store_by_game:
                rs_game.append(reward)
            else:
                all_rs.append(reward)

            episode_sum_reward += reward
            episode_length += 1
        if store_by_game:
            origin_obs_game = np.squeeze(np.array(origin_obs_game), axis=1)
            obs_game = np.squeeze(np.array(obs_game), axis=1)
            acs_game = np.squeeze(np.array(acs_game), axis=1)
            rs_game = np.squeeze(np.asarray(rs_game))
            all_orig_obs.append(origin_obs_game)
            all_obs.append(obs_game)
            all_acs.append(acs_game)
            all_rs.append(rs_game)
            if sample_parameters['store_code']:
                codes_game = np.squeeze(np.asarray(codes_game))
                all_codes.append(codes_game)

        sum_rewards.append(episode_sum_reward)
        lengths.append(episode_length)

    if store_by_game:
        if sample_parameters['store_code']:
            return all_orig_obs, all_obs, all_acs, all_rs, all_codes, sum_rewards, lengths
        else:
            return all_orig_obs, all_obs, all_acs, all_rs, sum_rewards, lengths
    else:
        all_orig_obs = np.squeeze(np.array(all_orig_obs), axis=1)
        all_obs = np.squeeze(np.array(all_obs), axis=1)
        all_acs = np.squeeze(np.array(all_acs), axis=1)
        all_rs = np.array(all_rs)
        sum_rewards = np.squeeze(np.array(sum_rewards), axis=1)
        lengths = np.array(lengths)
        if sample_parameters['store_code']:
            all_codes = np.squeeze(np.array(all_codes), axis=1)
            return all_orig_obs, all_obs, all_acs, all_rs, all_codes, sum_rewards, lengths
        else:
            return all_orig_obs, all_obs, all_acs, all_rs, sum_rewards, lengths
