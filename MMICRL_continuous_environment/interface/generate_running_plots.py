import os
import copy
import numpy as np
import sys

sys.path.append('../')
from utils.data_utils import read_running_logs_by_cid, compute_moving_average, mean_std_plot_results,read_running_logs
from utils.plot_utils import plot_curve, plot_shadow_curve


def plot_results(mean_results_moving_avg_dict,
                 std_results_moving_avg_dict,
                 ylim, label, method_names,
                 save_label,
                 legend_dict=None,
                 legend_size=12,
                 axis_size=None,
                 img_size=None,
                 title=None):
    plot_mean_y_dict = {}
    plot_std_y_dict = {}
    plot_x_dict = {}
    for method_name in method_names:
        plot_x_dict.update({method_name: [i for i in range(len(mean_results_moving_avg_dict[method_name]))]})
        plot_mean_y_dict.update({method_name: mean_results_moving_avg_dict[method_name]})
        print(method_name+'mean'+'\n')
        print(plot_mean_y_dict[method_name])
        plot_std_y_dict.update({method_name: std_results_moving_avg_dict[method_name]})
        print(method_name+'std'+'\n')
        print(plot_std_y_dict[method_name])
    plot_shadow_curve(draw_keys=method_names,
                      x_dict_mean=plot_x_dict,
                      y_dict_mean=plot_mean_y_dict,
                      x_dict_std=plot_x_dict,
                      y_dict_std=plot_std_y_dict,
                      img_size=img_size if img_size is not None else (6, 4.5),
                      ylim=ylim,
                      title=title,
                      xlabel='Episode',
                      ylabel=label,
                      legend_dict=legend_dict,
                      legend_size=legend_size,
                      axis_size=axis_size if axis_size is not None else 20,
                      title_size=24,
                      plot_name='./plot_results/{0}'.format(save_label), )


def generate_plots():
    last_num = 100

    env_id = 'HCWithPos-v0'
    method_names_labels_dict = {
        #"train_BCCL_HCWithPos-v0-multi_env": 'B2CL',
        #"train_ICRL_HCWithPos-v0-multi_env": 'MEICRL',
        #"train_InfoICRL_HCWithPos-v0-multi_env": 'InfoGAIL-ICRL',
        #"train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_den-multi_env": 'MMICRL-LD',
        "train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1-multi_env": 'MMICRL',
        #"train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_withGF-multi_env": 'GFICRL',
        #"train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_withGF_LSTM-multi_env": 'GFICRL_LSTM',
    #     "train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_npv30-multi_env": 'MM-ICRL-PV30',
    #     "train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_npv20-multi_env": 'MM-ICRL-PV20',
    #     "train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_npv10-multi_env": 'MM-ICRL-PV10',
    #    "train_me_c-0_ppo_lag_HCWithPos-v0-multi_env": 'PPO-Lag-v0',
    #    "train_me_c-1_ppo_lag_HCWithPos-v0-multi_env": 'PPO-Lag-v1',
    }

    #env_id = 'AntWall-v0'
    #method_names_labels_dict = {
    #     "train_BCCL_AntWall-v0-multi_env": 'B2CL',
    #     "train_ICRL_AntWall-v0-multi_env": 'MEICRL',
    #     "train_InfoICRL_AntWall-v0-multi_env": 'InfoGAIL-ICRL',
    #     "train_MEICRL_AntWall-v0_initden_den-multi_env": 'MMICRL-LD',
    #     "train_MEICRL_AntWall-v0_initden-multi_env": 'MMICRL',
        # "train_MEICRL_AntWall-v0_initden_withGF-multi_env": 'GFICRL',
        #"train_MEICRL_AntWall-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1-multi_env":'MEICRL_weight-5e-1',
    #      "train_me_c-0_ppo_lag_AntWall-v0-multi_env": 'PPO-Lag-v0',
    #      "train_me_c-1_ppo_lag_AntWall-v0-multi_env": 'PPO-Lag-v1',
    #}

    # env_id = 'InvertedPendulumWall-v0'
    # method_names_labels_dict = {
          #"train_MEICRL_Pendulum-v0-multi_env": 'MEICRL',
    #       "train_MEICRL_Pendulum-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1-multi_env":'MEICRL_weight-5e-1',
    #       "train_me_c-0_ppo_lag_Pendulum-v0-multi_env": 'PPO-Lag-v0',
    #       "train_me_c-1_ppo_lag_Pendulum-v0-multi_env": 'PPO-Lag-v1',
    # }

    #env_id = 'WalkerWithPos-v0'
    #method_names_labels_dict = {
    #     "train_BCCL_Walker-v0-multi_env": 'B2CL',
    #     "train_ICRL_Walker-v0-multi_env": 'MEICRL',
    #     "train_InfoICRL_Walker-v0-multi_env": 'InfoGAIL-ICRL',
    #     "train_MEICRL_Walker-v0_den-multi_env": 'MMICRL-LD',
    #    "train_MEICRL_Walker-v0-multi_env": 'MMICRL',
        
        #"train_MEICRL_Walker-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1-multi_env":'MEICRL_weight-5e-1',
        #"train_me_c-0_ppo_lag_Walker-v0-multi_env": 'PPO-Lag-v0',
        #"train_me_c-1_ppo_lag_Walker-v0-multi_env": 'PPO-Lag-v1',
    #}

    #env_id = 'SwimmerWithPos-v0'
    #method_names_labels_dict = {  
        #"train_BCCL_Swimmer-v0-multi_env": 'B2CL',
        #"train_ICRL_Swimmer-v0-multi_env": 'MEICRL',
        #"train_InfoICRL_Swimmer-v0-multi_env": 'InfoGAIL-ICRL',
        #"train_MEICRL_Swimmer-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_den-multi_env": 'MMICRL-LD',
        #"train_MEICRL_Swimmer-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1-multi_env": 'MMICRL',
        #"train_me_c-0_ppo_lag_Swimmer-v0-multi_env": 'PPO-Lag-v0',
        #"train_me_c-1_ppo_lag_Swimmer-v0-multi_env": 'PPO-Lag-v1',
    #}

    modes = ['train']
    plot_mode = 'all_Sanity'
    img_size = None
    axis_size = None
    if plot_mode == 'part':
        for method_name in method_names_labels_dict.copy().keys():
            if method_names_labels_dict[method_name] != 'PPO-Lag-v0' and method_names_labels_dict[method_name] != 'PPO-Lag-v1':
                del method_names_labels_dict[method_name]
    else:
        method_names_labels_dict = method_names_labels_dict
    for mode in modes:
        if env_id == 'HCWithPos-v0':
            max_episodes = 4000
            average_num = 100
            max_reward = 10000
            min_reward = -10000
            aid_num = 2
            plot_key = ['reward', 'reward_nc', 'constraint']
            label_key = ['reward', 'Feasible Cumulative Rewards', 'Constraint Violation Rate']
            plot_y_lim_dict = {'reward': None,
                               'reward_nc': None,
                               'constraint': None}
            constraint_keys = ['constraint']
            title = 'Blocked Half-Cheetah'
            log_path_dict = {
                "train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1-multi_env": [
                    #'../save_model/MEICRL_HCWithPos-v0/train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1-multi_env-Mar-13-2023-21_49-seed_123/',
                    #'../save_model/MEICRL_HCWithPos-v0/train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_npv30-multi_env-Apr-20-2023-15_01-seed_666/',
                    #'../save_model/MEICRL_HCWithPos-v0/train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1-multi_env-Apr-12-2023-02_00-seed_321/',
                    '../save_model/MEICRL_HCWithPos-v0/train_MEICRL_HCWithPos-v0-multi_env-May-22-2023-13_20-seed_123/',
                    '../save_model/MEICRL_HCWithPos-v0/train_MEICRL_HCWithPos-v0-multi_env-May-22-2023-03_26-seed_321/',
                    '../save_model/MEICRL_HCWithPos-v0/train_MEICRL_HCWithPos-v0-multi_env-May-21-2023-17_15-seed_666/',
                ],
                "train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_withGF-multi_env": [
                    '../save_model/MEICRL_HCWithPos-v0/train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_withGF-multi_env-Mar-28-2023-17_09-seed_123/',
                    #'../save_model/MEICRL_HCWithPos-v0/train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-1e-1_withGF-multi_env-Apr-03-2023-09_56-seed_123/',
                    #'../save_model/MEICRL_HCWithPos-v0/train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_withGF-multi_env-Mar-31-2023-13_44-seed_30/',
                ],
                "train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_withGF_LSTM-multi_env": [
                    '../save_model/MEICRL_HCWithPos-v0/train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_withGF-multi_env-Mar-29-2023-21_44-seed_123/',
                ],
                "train_me_c-0_ppo_lag_HCWithPos-v0-multi_env": [
                    '../save_model/PPO-Lag-HC/train_me_c-0_ppo_lag_HCWithPos-v0-multi_env-Jul-21-2022-11_52-seed_123/'
                ],
                "train_me_c-1_ppo_lag_HCWithPos-v0-multi_env": [
                    '../save_model/PPO-Lag-HC/train_me_c-1_ppo_lag_HCWithPos-v0-multi_env-Jul-21-2022-13_20-seed_123/'
                ],
                "train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_den-multi_env": [
                    '../save_model/MEICRL_HCWithPos-v0/train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_den-multi_env-Apr-01-2023-04_46-seed_321/',
                    '../save_model/MEICRL_HCWithPos-v0/train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_den-multi_env-Apr-17-2023-23_00-seed_123/',
                    '../save_model/MEICRL_HCWithPos-v0/train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_den-multi_env-Apr-01-2023-14_52-seed_666/',
                ],
                "train_InfoICRL_HCWithPos-v0-multi_env": [
                    #'../../constraint-learning-info/save_model/InfoICRL_HCWithPos-v0/train_InfoICRL_HCWithPos-v0-multi_env-Apr-23-2023-19:27-seed_123/',
                    #'../../constraint-learning-info/save_model/InfoICRL_HCWithPos-v0/train_InfoICRL_HCWithPos-v0-multi_env-Apr-24-2023-05:03-seed_321/',
                    #'../../constraint-learning-info/save_model/InfoICRL_HCWithPos-v0/train_InfoICRL_HCWithPos-v0-multi_env-Apr-24-2023-11:07-seed_666/',
                    #'../../constraint-learning-info/save_model/InfoICRL_HCWithPos-v0/train_InfoICRL_HCWithPos-v0-multi_env-Apr-14-2023-17:05-seed_123/',
                    '../../constraint-learning-info/save_model/InfoICRL_HCWithPos-v0/train_InfoICRL_HCWithPos-v0-multi_env-May-02-2023-00:35-seed_123/',
                    '../../constraint-learning-info/save_model/InfoICRL_HCWithPos-v0/train_InfoICRL_HCWithPos-v0-multi_env-May-02-2023-11:37-seed_321/',
                    '../../constraint-learning-info/save_model/InfoICRL_HCWithPos-v0/train_InfoICRL_HCWithPos-v0-multi_env-May-02-2023-20:29-seed_666/',
                ],
                "train_ICRL_HCWithPos-v0-multi_env": [
                    '../save_model/ICRL_HCWithPos-v0/train_ICRL_HCWithPos-v0-multi_env-Apr-24-2023-13_11-seed_123/',
                    '../save_model/ICRL_HCWithPos-v0/train_ICRL_HCWithPos-v0-multi_env-Apr-25-2023-09_08-seed_321/',
                    #'../save_model/ICRL_HCWithPos-v0/train_ICRL_HCWithPos-v0-multi_env-Apr-21-2023-06_47-seed_321/',
                    #'../save_model/ICRL_HCWithPos-v0/train_ICRL_HCWithPos-v0-multi_env-Apr-21-2023-18_12-seed_666/',
                ],
                "train_BCCL_HCWithPos-v0-multi_env": [
                    '../save_model/Binary_HCWithPos-v0/train_Binary_HCWithPos-v0-multi_env-Apr-27-2023-02_19-seed_321/',
                    '../save_model/Binary_HCWithPos-v0/train_Binary_HCWithPos-v0-multi_env-Apr-27-2023-18_22-seed_123/',
                    '../save_model/Binary_HCWithPos-v0/train_Binary_HCWithPos-v0-multi_env-Apr-28-2023-10_00-seed_666/',
                ],
                "train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_npv30-multi_env": [
                    '../save_model/MEICRL_HCWithPos-v0/train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_npv30-multi_env-Apr-20-2023-15_01-seed_666/',
                ],
                "train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_npv10-multi_env": [
                    '../save_model/MEICRL_HCWithPos-v0/train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1-multi_env-Mar-10-2023-14:28-seed_666/',
                ],
                "train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_npv20-multi_env": [
                    '../save_model/MEICRL_HCWithPos-v0/train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_npv20-multi_env-Apr-24-2023-14_05-seed_666/',
                ],
            }
        elif env_id == 'AntWall-v0':
            max_episodes = 15000
            average_num = 300
            max_reward = 5000
            min_reward = -5000
            aid_num = 2
            plot_key = ['reward', 'reward_nc', 'constraint']
            label_key = ['reward', 'Feasible Cumulative Rewards', 'Constraint Violation Rate']
            plot_y_lim_dict = {'reward': None,
                               'reward_nc': None,
                               'constraint': None}
            constraint_keys = ['constraint']
            title = 'Blocked Antwall'
            log_path_dict = {
                "train_MEICRL_AntWall-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1-multi_env": [
                    #'../save_model/MEICRL_AntWall-v0/train_MEICRL_AntWall-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1-multi_env-Mar-13-2023-11_27-seed_123/',
                ],
                "train_MEICRL_AntWall-v0_initden-multi_env": [
                    #'../save_model/MEICRL_AntWall-v0/train_MEICRL_AntWall-v0_initden-multi_env-Apr-01-2023-03_04-seed_200/',
                    #'../save_model/MEICRL_AntWall-v0/train_MEICRL_AntWall-v0_initden_npv50-multi_env-Apr-26-2023-15_45-seed_666/',
                    #'../save_model/MEICRL_AntWall-v0/train_MEICRL_AntWall-v0_initden_npv50-multi_env-May-01-2023-17_22-seed_321/',
                    #'../save_model/MEICRL_AntWall-v0/train_MEICRL_AntWall-v0_initden_npv50-multi_env-May-01-2023-01_28-seed_123/',
                    '../save_model/MEICRL_AntWall-v0/train_MEICRL_AntWall-v0_initden_npv50-multi_env-May-12-2023-17_21-seed_321/',
                    '../save_model/MEICRL_AntWall-v0/train_MEICRL_AntWall-v0_initden_npv50-multi_env-May-11-2023-10_39-seed_123/',
                    '../save_model/MEICRL_AntWall-v0/train_MEICRL_AntWall-v0_initden_npv50-multi_env-May-11-2023-21_28-seed_666/',
                ],
                "train_MEICRL_AntWall-v0_initden_withGF-multi_env": [
                    '../save_model/MEICRL_AntWall-v0/train_MEICRL_AntWall-v0_initden_withGF-multi_env-Apr-04-2023-22_15-seed_200/',
                ],
                "train_MEICRL_AntWall-v0_initden_den-multi_env": [
                    '../save_model/MEICRL_AntWall-v0/train_MEICRL_AntWall-v0_initden-multi_env-Apr-05-2023-01_52-seed_123/',
                    '../save_model/MEICRL_AntWall-v0/train_MEICRL_AntWall-v0_initden_den-multi_env-Apr-23-2023-00_28-seed_321/',
                    '../save_model/MEICRL_AntWall-v0/train_MEICRL_AntWall-v0_initden_den_npv50-multi_env-May-03-2023-09_56-seed_666/',
                ],
                "train_ICRL_AntWall-v0-multi_env": [
                    '../save_model/ICRL_AntWall-v0/train_ICRL_AntWall-v0-multi_env-Apr-24-2023-13_17-seed_123/',
                    '../save_model/ICRL_AntWall-v0/train_ICRL_AntWall-v0-multi_env-Apr-25-2023-10_50-seed_321/',
                    '../save_model/ICRL_AntWall-v0/train_ICRL_AntWall-v0-multi_env-Apr-26-2023-05_15-seed_666/',
                    #'../save_model/ICRL_AntWall-v0/train_ICRL_AntWall-v0-multi_env-Apr-21-2023-04_33-seed_321/'
                ],
                "train_BCCL_AntWall-v0-multi_env": [
                    '../save_model/Binary_AntWall-v0/train_Binary_AntWall-v0-multi_env-May-06-2023-10_09-seed_123/',
                    '../save_model/Binary_AntWall-v0/train_Binary_AntWall-v0-multi_env-May-06-2023-22_06-seed_321/',
                    '../save_model/Binary_AntWall-v0/train_Binary_AntWall-v0-multi_env-May-07-2023-09_18-seed_666/',
                    #'../save_model/ICRL_AntWall-v0/train_ICRL_AntWall-v0-multi_env-Apr-21-2023-04_33-seed_321/'
                ],
                "train_InfoICRL_AntWall-v0-multi_env": [
                    '../../constraint-learning-info/save_model/InfoICRL_AntWall-v0/train_InfoICRL_AntWall-v0-multi_env-May-02-2023-09:18-seed_123/',
                    '../../constraint-learning-info/save_model/InfoICRL_AntWall-v0/train_InfoICRL_AntWall-v0-multi_env-May-03-2023-07:21-seed_321/',
                    '../../constraint-learning-info/save_model/InfoICRL_AntWall-v0/train_InfoICRL_AntWall-v0-multi_env-May-04-2023-06:33-seed_666/',
                    #'../save_model/InfoICRL_AntWall-v0/train_InfoICRL_AntWall-v0-multi_env-Apr-14-2023-21_03-seed_321/',
                    #'../save_model/InfoICRL_AntWall-v0/train_InfoICRL_AntWall-v0-multi_env-Apr-15-2023-00_36-seed_666/',
                ],
                "train_me_c-0_ppo_lag_AntWall-v0-multi_env": [
                    '../save_model/PPO-Lag-AntWall/train_me_c-0_ppo_lag_AntWall-v0-multi_env-Mar-02-2023-20:55-seed_123/'
                ],
                "train_me_c-1_ppo_lag_AntWall-v0-multi_env": [
                    '../save_model/PPO-Lag-AntWall/train_me_c-1_ppo_lag_AntWall-v0-multi_env-Mar-03-2023-01:23-seed_123/'
                ],
            }
        elif env_id == 'InvertedPendulumWall-v0':
            max_episodes = 80000
            average_num = 2000
            max_reward = 10000
            min_reward = -10000
            aid_num = 2
            plot_key = ['reward', 'reward_nc', 'constraint']
            label_key = ['reward', 'reward_nc', 'Constraint Violation Rate']
            constraint_keys = ['constraint']
            plot_y_lim_dict = {'reward': None,
                               'reward_nc': None,
                               'constraint': None}
            title = 'Blocked InvertedPendulumWall'
            log_path_dict = {
                "train_MEICRL_Pendulum-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1-multi_env": [
                    '../save_model/MEICRL_InvertedPendulumWall-v0/train_MEICRL_Pendulum-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1-multi_env-Mar-26-2023-20_37-seed_123/',
                    '../save_model/MEICRL_InvertedPendulumWall-v0/train_MEICRL_Pendulum-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1-multi_env-Mar-15-2023-19_40-seed_321/',
                ],
                "train_MEICRL_Pendulum-v0-multi_env": [
                    #'../save_model/MEICRL_InvertedPendulumWall-v0/train_MEICRL_Pendulum-v0-multi_env-Mar-09-2023-17:09-seed_123/',
                ],
                "train_me_c-0_ppo_lag_Pendulum-v0-multi_env": [
                    '../save_model/PPO-Lag-Pendulum/train_me_c-0_ppo_lag_Pendulum-v0-multi_env-Mar-05-2023-23:59-seed_123/'
                ],
                "train_me_c-1_ppo_lag_Pendulum-v0-multi_env": [
                    '../save_model/PPO-Lag-Pendulum/train_me_c-1_ppo_lag_Pendulum-v0-multi_env-Mar-06-2023-19:19-seed_123/'
                ],
            }
        elif env_id == 'SwimmerWithPos-v0':
            max_episodes = 15000
            average_num = 300
            max_reward = 10000
            min_reward = -10000
            aid_num = 2
            plot_key = ['reward', 'reward_nc', 'constraint']
            label_key = ['reward', 'Feasible Cumulative Rewards', 'Constraint Violation Rate']
            constraint_keys = ['constraint']
            plot_y_lim_dict = {'reward': None,
                               'reward_nc': None,
                               'constraint': None}
            title = 'Blocked SwimmerWithPos'
            log_path_dict = {
                "train_MEICRL_Swimmer-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1-multi_env": [
                    #'../save_model/MEICRL_HCWithPos-v0/sanity_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2-multi_env-Nov-04-2022-17:05-seed_123/',
                    #'../save_model/MEICRL_SwimmerWithPos-v0/train_MEICRL_Swimmer-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1-multi_env-Mar-14-2023-13_08-seed_666/',
                    ##'../save_model/MEICRL_SwimmerWithPos-v0/train_MEICRL_Swimmer-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_newdata1-multi_env-Apr-10-2023-11_08-seed_666/',
                    ##'../save_model/MEICRL_SwimmerWithPos-v0/train_MEICRL_Swimmer-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_newdata1-multi_env-Apr-10-2023-11_08-seed_321/',
                    ##'../save_model/MEICRL_SwimmerWithPos-v0/train_MEICRL_Swimmer-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_newdata1-multi_env-Apr-18-2023-15_48-seed_123/',
                    #'../save_model/MEICRL_SwimmerWithPos-v0/train_MEICRL_Swimmer-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_newdata2-multi_env-Apr-14-2023-16_37-seed_666/',
                    #'../save_model/MEICRL_SwimmerWithPos-v0/train_MEICRL_Swimmer-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_newdata2-multi_env-Apr-16-2023-13_06-seed_321/',
                    '../save_model/MEICRL_SwimmerWithPos-v0/train_MEICRL_Swimmer-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_newdata1-multi_env-May-14-2023-19_27-seed_321/',
                    '../save_model/MEICRL_SwimmerWithPos-v0/train_MEICRL_Swimmer-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_newdata1-multi_env-May-12-2023-17_46-seed_666/',
                    '../save_model/MEICRL_SwimmerWithPos-v0/train_MEICRL_Swimmer-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_newdata1-multi_env-May-15-2023-15_54-seed_123/',
                    #'../save_model/MEICRL_SwimmerWithPos-v0/train_MEICRL_Swimmer-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_newdata1-multi_env-May-11-2023-16_28-seed_666/',
                    #'../save_model/MEICRL_SwimmerWithPos-v0/train_MEICRL_Swimmer-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_newdata1-multi_env-May-13-2023-19_33-seed_321/',
                    #'../save_model/MEICRL_SwimmerWithPos-v0/train_MEICRL_Swimmer-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_newdata1-multi_env-May-15-2023-15_54-seed_123/',
                ],
                "train_InfoICRL_Swimmer-v0-multi_env": [
                    '../../constraint-learning-info/save_model/InfoICRL_Swimmer-v0/train_InfoICRL_Swimmer-v0-multi_env-May-03-2023-09:43-seed_123/',
                    '../../constraint-learning-info/save_model/InfoICRL_Swimmer-v0/train_InfoICRL_Swimmer-v0-multi_env-May-04-2023-11:39-seed_321/',
                    #'../../constraint-learning-info/save_model/InfoICRL_Swimmer-v0/train_InfoICRL_Swimmer-v0-multi_env-Apr-19-2023-19:32-seed_666/',
                ],
                "train_ICRL_Swimmer-v0-multi_env": [
                    '../save_model/ICRL_SwimmerWithPos-v0/train_ICRL_Swimmer-v0-multi_env-Apr-25-2023-16_14-seed_123/',
                    '../save_model/ICRL_SwimmerWithPos-v0/train_ICRL_Swimmer-v0-multi_env-Apr-28-2023-12_40-seed_321/',
                    '../save_model/ICRL_SwimmerWithPos-v0/train_ICRL_Swimmer-v0-multi_env-May-01-2023-06_26-seed_666/',
                ],
                "train_BCCL_Swimmer-v0-multi_env": [
                    '../save_model/Binary_SwimmerWithPos-v0/train_Binary_Swimmer-v0-multi_env-May-08-2023-04_44-seed_123/',
                    '../save_model/Binary_SwimmerWithPos-v0/train_Binary_Swimmer-v0-multi_env-May-04-2023-13_47-seed_666/',
                    '../save_model/Binary_SwimmerWithPos-v0/train_Binary_Swimmer-v0-multi_env-May-06-2023-10_09-seed_321/',
                ],
                "train_MEICRL_Swimmer-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_den-multi_env": [
                    '../save_model/MEICRL_SwimmerWithPos-v0/train_MEICRL_Swimmer-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_newdata1_den-multi_env-Apr-20-2023-02_27-seed_123/',
                    '../save_model/MEICRL_SwimmerWithPos-v0/train_MEICRL_Swimmer-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_newdata1_den-multi_env-Apr-22-2023-02_40-seed_666/',
                    '../save_model/MEICRL_SwimmerWithPos-v0/train_MEICRL_Swimmer-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_newdata1_den-multi_env-Apr-17-2023-20_39-seed_321/',
                ],
                "train_me_c-0_ppo_lag_Swimmer-v0-multi_env": [
                    '../save_model/PPO-Lag-Swimmer/train_me_c-0_ppo_lag_Swimmer-v0-multi_env-Mar-03-2023-15:15-seed_123/'
                ],
                "train_me_c-1_ppo_lag_Swimmer-v0-multi_env": [
                    '../save_model/PPO-Lag-Swimmer/train_me_c-1_ppo_lag_Swimmer-v0-multi_env-Mar-03-2023-20:42-seed_123/'
                ],
            }
        elif env_id == 'WalkerWithPos-v0':
            max_episodes = 40000
            average_num = 100
            max_reward = 10000
            min_reward = -10000
            aid_num = 2
            plot_key = ['reward', 'reward_nc', 'constraint']
            label_key = ['reward', 'Feasible Cumulative Rewards', 'Constraint Violation Rate']
            plot_y_lim_dict = {'reward': None,
                               'reward_nc': None,
                               'constraint': None}
            constraint_keys = ['constraint']
            title = 'Blocked WalkerWithPos'
            log_path_dict = {
                "train_MEICRL_Walker-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1-multi_env": [
                    #'../save_model/MEICRL_HCWithPos-v0/sanity_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2-multi_env-Nov-04-2022-17:05-seed_123/',
                    '../save_model/MEICRL_WalkerWithPos-v0/train_MEICRL_Walker-v0-multi_env-Apr-01-2023-21_58-seed_123/',
                    
                ],
                "train_MEICRL_Walker-v0-multi_env": [
                    #'../save_model/MEICRL_WalkerWithPos-v0/train_MEICRL_Walker-v0_npv30-multi_env-May-06-2023-10_15-seed_321/',
                    #'../save_model/MEICRL_WalkerWithPos-v0/train_MEICRL_Walker-v0-multi_env-May-06-2023-12_11-seed_666/',
                    #'../save_model/MEICRL_WalkerWithPos-v0/train_MEICRL_Walker-v0-multi_env-Apr-03-2023-14_00-seed_200/',
                    ##'../save_model/MEICRL_WalkerWithPos-v0/train_MEICRL_Walker-v0_npv30-multi_env-May-02-2023-00_52-seed_123/',
                    #'../save_model/MEICRL_WalkerWithPos-v0/train_MEICRL_Walker-v0-multi_env-May-04-2023-15_56-seed_123/',
                    #'../save_model/MEICRL_WalkerWithPos-v0/train_MEICRL_Walker-v0_npv50-multi_env-May-16-2023-18_34-seed_666/',
                    #'../save_model/MEICRL_WalkerWithPos-v0/train_MEICRL_Walker-v0_npv50-multi_env-May-14-2023-23_16-seed_123/',
                    #'../save_model/MEICRL_WalkerWithPos-v0/train_MEICRL_Walker-v0_npv50-multi_env-May-13-2023-02_51-seed_321/',
                    #'../save_model/MEICRL_WalkerWithPos-v0/train_MEICRL_Walker-v0-multi_env-May-14-2023-23_19-seed_200/',
                    #'../save_model/MEICRL_WalkerWithPos-v0/train_MEICRL_Walker-v0_npv50-multi_env-May-15-2023-16_08-seed_123/',
                    #'../save_model/MEICRL_WalkerWithPos-v0/train_MEICRL_Walker-v0_npv50-multi_env-May-13-2023-23_41-seed_321/',
                    '../save_model/MEICRL_WalkerWithPos-v0/train_MEICRL_Walker-v0_npv30-multi_env-May-12-2023-23_29-seed_321/',
                    '../save_model/MEICRL_WalkerWithPos-v0/train_MEICRL_Walker-v0_npv30-multi_env-May-16-2023-20_47-seed_123/',
                    '../save_model/MEICRL_WalkerWithPos-v0/train_MEICRL_Walker-v0_npv30-multi_env-May-21-2023-17_16-seed_666/',
                ],
                "train_BCCL_Walker-v0-multi_env": [
                    '../save_model/Binary_WalkerWithPos-v0/train_Binary_Walker-v0-multi_env-May-06-2023-17_27-seed_123/',
                    '../save_model/Binary_WalkerWithPos-v0/train_Binary_Walker-v0-multi_env-May-07-2023-18_34-seed_321/',
                    '../save_model/Binary_WalkerWithPos-v0/train_Binary_Walker-v0-multi_env-May-08-2023-20_32-seed_666/',
                ],
                "train_ICRL_Walker-v0-multi_env": [
                    #'../save_model/ICRL_WalkerWithPos-v0/train_ICRL_Walker-v0-multi_env-May-06-2023-17_44-seed_123/',
                    '../save_model/ICRL_WalkerWithPos-v0/train_ICRL_Walker-v0-multi_env-May-08-2023-23_34-seed_321/',
                    '../save_model/ICRL_WalkerWithPos-v0/train_ICRL_Walker-v0-multi_env-May-10-2023-19_43-seed_666/',
                    '../save_model/ICRL_WalkerWithPos-v0/train_ICRL_Walker-v0-multi_env-May-07-2023-11_00-seed_123/',
                ],
                "train_InfoICRL_Walker-v0-multi_env": [
                    '../../constraint-learning-info/save_model/InfoICRL_Walker-v0/train_InfoICRL_Walker-v0-multi_env-May-06-2023-12:03-seed_123/',
                    '../../constraint-learning-info/save_model/InfoICRL_Walker-v0/train_InfoICRL_Walker-v0-multi_env-May-07-2023-06:38-seed_321/',
                    '../../constraint-learning-info/save_model/InfoICRL_Walker-v0/train_InfoICRL_Walker-v0-multi_env-May-07-2023-23:57-seed_666/',
                ],
                "train_me_c-0_ppo_lag_Walker-v0-multi_env": [
                    '../save_model/PPO-Lag-Walker/train_me_c-0_ppo_lag_Walker-v0-multi_env-Mar-21-2023-13:02-seed_123/'
                ],
                "train_me_c-1_ppo_lag_Walker-v0-multi_env": [
                    '../save_model/PPO-Lag-Walker/train_me_c-1_ppo_lag_Walker-v0-multi_env-Mar-21-2023-17:51-seed_123/'
                ],
                "train_MEICRL_Walker-v0_den-multi_env": [
                    '../save_model/MEICRL_WalkerWithPos-v0/train_MEICRL_Walker-v0_den-multi_env-Apr-19-2023-08_23-seed_666/',
                    #'../save_model/MEICRL_WalkerWithPos-v0/train_MEICRL_Walker-v0_den-multi_env-May-08-2023-11_35-seed_666/',
                    '../save_model/MEICRL_WalkerWithPos-v0/train_MEICRL_Walker-v0_den-multi_env-Apr-17-2023-20_25-seed_321/',
                    '../save_model/MEICRL_WalkerWithPos-v0/train_MEICRL_Walker-v0_den-multi_env-May-08-2023-11_36-seed_321/',
                    '../save_model/MEICRL_WalkerWithPos-v0/train_MEICRL_Walker-v0_den-multi_env-Apr-18-2023-17_51-seed_123/', 
                    #'../save_model/MEICRL_WalkerWithPos-v0/train_MEICRL_Walker-v0_den-multi_env-Apr-16-2023-17_25-seed_200/',
                    #'../save_model/MEICRL_WalkerWithPos-v0/train_MEICRL_Walker-v0_den-multi_env-Apr-12-2023-13_47-seed_200/',
                    #'../save_model/MEICRL_WalkerWithPos-v0/train_MEICRL_Walker-v0_den-multi_env-May-08-2023-11_35-seed_666/',
                    #'../save_model/MEICRL_WalkerWithPos-v0/train_MEICRL_Walker-v0_den-multi_env-May-08-2023-11_36-seed_321/',
                ],
            }
        else:
            raise ValueError("Unknown env id {0}".format(env_id))

        all_mean_dict_by_cid = {}
        all_std_dict_by_cid = {}
        all_mean_dict = {}        
        all_std_dict = {}
        all_results = []
        for aid in range(aid_num):
            all_mean_dict_by_cid.update({aid: {}})
            all_std_dict_by_cid.update({aid: {}})
        for method_name in method_names_labels_dict.keys():
          if 'ppo_lag' not in method_name:
            all_results_by_cid = {}
            for aid in range(aid_num):
                all_results_by_cid.update({aid: []})
            for log_path in log_path_dict[method_name]:
                monitor_path_all = []
                if mode == 'train':
                    run_files = os.listdir(log_path)
                    for file in run_files:
                        if 'monitor' in file:
                            monitor_path_all.append(log_path + file)
                else:
                    monitor_path_all.append(log_path + 'test/test.monitor.csv')

                results_by_cid = read_running_logs_by_cid(monitor_path_all=monitor_path_all, read_keys=plot_key,
                                                          max_episodes=(max_episodes + float(
                                                              max_episodes / 5)) * aid_num,
                                                          max_reward=max_reward, min_reward=min_reward, cid_num=aid_num)
                for aid in range(aid_num):
                    all_results_by_cid[aid].append(results_by_cid[aid])

            for aid in range(aid_num):
                mean_dict, std_dict = mean_std_plot_results(all_results_by_cid[aid])
                all_mean_dict_by_cid[aid].update({method_name: {}})
                all_std_dict_by_cid[aid].update({method_name: {}})

                if not os.path.exists(os.path.join('./plot_results/', env_id)):
                    os.mkdir(os.path.join('./plot_results/', env_id))
                if not os.path.exists(os.path.join('./plot_results/', env_id, method_name)):
                    os.mkdir(os.path.join('./plot_results/', env_id, method_name))

                for idx in range(len(plot_key)):
                    mean_results_moving_average = compute_moving_average(result_all=mean_dict[plot_key[idx]],
                                                                         average_num=average_num)
                    std_results_moving_average = compute_moving_average(result_all=std_dict[plot_key[idx]],
                                                                        average_num=average_num)
                    if max_episodes:
                        mean_results_moving_average = mean_results_moving_average[:max_episodes]
                        std_results_moving_average = std_results_moving_average[:max_episodes]
                    all_mean_dict_by_cid[aid][method_name].update({plot_key[idx]: mean_results_moving_average})
                    all_std_dict_by_cid[aid][method_name].update({plot_key[idx]: std_results_moving_average / 2})
                    plot_results(mean_results_moving_avg_dict={method_name: mean_results_moving_average},
                                 std_results_moving_avg_dict={method_name: std_results_moving_average},
                                 label=plot_key[idx],
                                 method_names=[method_name],
                                 ylim=plot_y_lim_dict[plot_key[idx]],
                                 save_label=os.path.join(env_id, method_name,
                                                         plot_key[idx] + '_c{0}_'.format(aid) + '_' + mode),
                                 title=title+' c_'+str(aid),
                                 axis_size=axis_size,
                                 img_size=img_size,
                                 )
          else:
            
            for log_path in log_path_dict[method_name]:
                    monitor_path_all = []
                    if mode == 'train':
                        run_files = os.listdir(log_path)
                        for file in run_files:
                            if 'monitor' in file:
                                monitor_path_all.append(log_path + file)
                    else:
                        monitor_path_all.append(log_path + 'test/test.monitor.csv')
                    #if (method_names_labels_dict[method_name] == "PPO-Lag-v1" or
                    #    method_names_labels_dict[method_name] == "PPO-Lag-v0" or
                    #    method_names_labels_dict[method_name] == "PI-Lag") and plot_mode != "part":
                          #if 'reward_nc' in plot_key:
                              #plot_key[plot_key.index('reward_nc')] = 'reward'
                    results, valid_rewards, valid_episodes = read_running_logs(monitor_path_all=monitor_path_all,
                                                                               read_keys=plot_key,
                                                                               max_reward=max_reward,
                                                                               min_reward=min_reward,
                                                                               max_episodes=max_episodes,
                                                                               constraint_keys=constraint_keys)
                    if (method_names_labels_dict[method_name] == "PPO-Lag-v1" or
                        method_names_labels_dict[method_name] == "PPO-Lag-v0" or
                        method_names_labels_dict[method_name] == "PI-Lag") and plot_mode != "part":
                        results_copy_ = copy.copy(results)
                        for key in results.keys():
                            fill_value = np.mean(results_copy_[key][-last_num:])
                            results[key] = [fill_value for item in range(max_episodes + 100)]

                    all_results.append(results)

            mean_dict, std_dict = mean_std_plot_results(all_results)
            all_mean_dict.update({method_name: {}})
            all_std_dict.update({method_name: {}})

            if not os.path.exists(os.path.join('./plot_results/', env_id)):
                os.mkdir(os.path.join('./plot_results/', env_id))
            if not os.path.exists(os.path.join('./plot_results/', env_id, method_name)):
                os.mkdir(os.path.join('./plot_results/', env_id, method_name))

            for idx in range(len(plot_key)):
                mean_results_moving_average = compute_moving_average(result_all=mean_dict[plot_key[idx]],
                                                                     average_num=average_num)
                std_results_moving_average = compute_moving_average(result_all=std_dict[plot_key[idx]],
                                                                    average_num=average_num)
                if max_episodes:
                    mean_results_moving_average = mean_results_moving_average[:max_episodes]
                    std_results_moving_average = std_results_moving_average[:max_episodes]
                all_mean_dict[method_name].update({plot_key[idx]: mean_results_moving_average})
                if (method_names_labels_dict[method_name] == "PPO-Lag-v0" or
                    method_names_labels_dict[method_name] == "PPO-Lag-v1") and plot_mode != "part":
                    all_std_dict[method_name].update({plot_key[idx]: np.zeros(std_results_moving_average.shape)})
                else:
                    all_std_dict[method_name].update({plot_key[idx]: std_results_moving_average / 2})

                plot_results(mean_results_moving_avg_dict={method_name: mean_results_moving_average},
                             std_results_moving_avg_dict={method_name: std_results_moving_average},
                             label=plot_key[idx],
                             method_names=[method_name],
                             ylim=plot_y_lim_dict[plot_key[idx]],
                             save_label=os.path.join(env_id, method_name, plot_key[idx] + '_' + mode),
                             title=title,
                             axis_size=axis_size,
                             img_size=img_size,
                             )

        
        for aid in range(aid_num):
            for idx in range(len(plot_key)):
                mean_results_moving_avg_dict = {}
                std_results_moving_avg_dict = {}
                for method_name in method_names_labels_dict.keys():
                    if 'c-0_ppo_lag' in method_name and aid==0:
                        mean_results_moving_avg_dict.update({method_name: all_mean_dict[method_name][plot_key[idx]]})
                        std_results_moving_avg_dict.update({method_name: all_std_dict[method_name][plot_key[idx]]})
                        continue
                    elif 'c-1_ppo_lag' in method_name and aid==1:
                        mean_results_moving_avg_dict.update({method_name: all_mean_dict[method_name][plot_key[idx]]})
                        std_results_moving_avg_dict.update({method_name: all_std_dict[method_name][plot_key[idx]]}) 
                        continue
                    elif 'c-1_ppo_lag' in method_name and aid==0:
                        continue
                    elif 'c-0_ppo_lag' in method_name and aid==1:
                        continue
                    mean_results_moving_avg_dict.update(
                        {method_name: all_mean_dict_by_cid[aid][method_name][plot_key[idx]]})
                    std_results_moving_avg_dict.update(
                        {method_name: all_std_dict_by_cid[aid][method_name][plot_key[idx]]})
                if aid == 0:
                    method_names=list(method_names_labels_dict.keys())
                    for method_name in method_names_labels_dict.keys():
                        if 'c-1_ppo_lag' in method_name:
                             method_names.remove(method_name)
                             break                    
                elif aid == 1:
                    method_names=list(method_names_labels_dict.keys())
                    for method_name in method_names_labels_dict.keys():
                        if 'c-0_ppo_lag' in method_name:
                             method_names.remove(method_name)
                             break
                plot_results(mean_results_moving_avg_dict=mean_results_moving_avg_dict,
                                 std_results_moving_avg_dict=std_results_moving_avg_dict,
                                 label=label_key[idx],
                                 method_names=method_names,
                                 ylim=plot_y_lim_dict[plot_key[idx]],
                                 save_label=os.path.join(env_id, plot_key[idx] + '_c{0}_'.format(aid) + '_'
                                                         + mode + '_' + env_id + '_' + plot_mode),
                                 # legend_size=18,
                                 legend_dict=method_names_labels_dict,
                                 title=title+' z_'+str(aid),
                                 axis_size=axis_size,
                                 img_size=img_size,
                                 )


if __name__ == "__main__":
    generate_plots()
