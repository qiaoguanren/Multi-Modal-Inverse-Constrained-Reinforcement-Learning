import os,sys

sys.path.append('../')
from utils.data_utils import read_running_logs_by_cid, compute_moving_average, mean_std_plot_results
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
        print(method_name+'\n')
        plot_mean_y_dict.update({method_name: mean_results_moving_avg_dict[method_name]})
        print('mean' + '\n')
        print(plot_mean_y_dict[method_name][-1])
        plot_std_y_dict.update({method_name: std_results_moving_avg_dict[method_name]})
        print('std' + '\n')
        print(plot_std_y_dict[method_name][-1])
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
                      axis_size=axis_size if axis_size is not None else 18,
                      title_size=24,
                      plot_name='./plot_results/{0}'.format(save_label), )


def generate_plots():
    env_id = 'HCWithPos-v0'
    method_names_labels_dict = {
        "sanity_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2-multi_env": 'sanity_MEICRL',
        "semi_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-1e-1_advloss-multi_env": 'semi_MEICRL_weight-1e-1_advloss',
        "semi_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-1e-1_closs-multi_env": 'semi_MEICRL_weight-1e-1_closs',
        "semi_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-1e-1-multi_env": 'semi_MEICRL_weight-1e-1',
        # "robust_check_0.3_train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-0_robust-3e-1_advloss-multi_env": 'robust_MEICRL_weight-0_robust-3e-1_advloss',
        # "robust_check_0.3_train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-1e-1_robust-3e-1_advloss-multi_env": 'robust_MEICRL_weight-1e-1_robust-3e-1_advloss',
        # "robust_check_0.3_train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-3e-1_robust-3e-1_advloss-multi_env": 'robust_MEICRL_weight-3e-1_robust-3e-1_advloss',
        # "robust_check_0.3_train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_robust-3e-1_advloss-multi_env": 'robust_MEICRL_weight-5e-1_robust-3e-1_advloss',
        # "robust_check_0.2_train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-0_robust-2e-1_advloss-multi_env": 'robust_MEICRL_weight-0_robust-2e-1_advloss',
        # "robust_check_0.2_train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-1e-1_robust-2e-1_advloss-multi_env": 'robust_MEICRL_weight-1e-1_robust-2e-1_advloss',
        # "robust_check_0.2_train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-3e-1_robust-2e-1_advloss-multi_env": 'robust_MEICRL_weight-3e-1_robust-2e-1_advloss',
        # "robust_check_0.2_train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_robust-2e-1_advloss-multi_env": 'robust_MEICRL_weight-5e-1_robust-2e-1_advloss',
        "robust_check_0.1_train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-0_robust-1e-1_advloss-multi_env": 'robust_MEICRL_weight-0_robust-1e-1_advloss',
        "robust_check_0.1_train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-1e-1_robust-1e-1_advloss-multi_env": 'robust_MEICRL_weight-1e-1_robust-1e-1_advloss',
        "robust_check_0.1_train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-3e-1_robust-1e-1_advloss-multi_env": 'robust_MEICRL_weight-3e-1_robust-1e-1_advloss',
        "robust_check_0.1_train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_robust-1e-1_advloss-multi_env": 'robust_MEICRL_weight-5e-1_robust-1e-1_advloss',
    }

    env_id = 'WGW-v0'
    method_names_labels_dict = {
        # 'train_Binary_WGW-v0': 'B2CL',
        # 'train_MEICRL_WGW-v0': 'MEICRL',
        # 'train_InfoGAIL-ICRL_WGW-v0': 'InfoGAIL-ICRL',
        # 'train_MMICRL-LD_WGW-v0': 'MMICRL-LD',
        #'train_MMICRL_WGW-v0': 'MMICRL',
        # 'train_Binary_WGW-v1': 'B2CL',
        # 'train_MEICRL_WGW-v1': 'MEICRL',
        # 'train_InfoGAIL-ICRL_WGW-v1': 'InfoGAIL-ICRL',
        # 'train_MMICRL-LD_WGW-v1': 'MMICRL-LD',
        #'train_MMICRL_WGW-v1': 'MMICRL',
        # 'train_Binary_WGW-v2': 'B2CL',
        # 'train_MEICRL_WGW-v2': 'MEICRL',
        # 'train_InfoGAIL-ICRL_WGW-v2': 'InfoGAIL-ICRL',
        # 'train_MMICRL-LD_WGW-v2': 'MMICRL-LD',
        #'train_MMICRL_WGW-v2': 'MMICRL',
        # 'train_Binary_WGW-v3': 'B2CL',
        # 'train_MEICRL_WGW-v3': 'MEICRL',
        # 'train_InfoGAIL-ICRL_WGW-v3': 'InfoGAIL-ICRL',
        # 'train_MMICRL-LD_WGW-v3': 'MMICRL-LD',
        'train_MMICRL_WGW-v3': 'MMICRL',
        # 'train_MMICRL_WGW-v4': 'MMICRL',
    }

    modes = ['train']
    plot_mode = 'all_Sanity'
    img_size = None
    axis_size = None
    if plot_mode == 'part':
        for method_name in method_names_labels_dict.copy().keys():
            if method_names_labels_dict[method_name] != 'PPO' and method_names_labels_dict[method_name] != 'PPO_lag':
                del method_names_labels_dict[method_name]
    else:
        method_names_labels_dict = method_names_labels_dict
    for mode in modes:
        if env_id == 'HCWithPos-v0':
            max_episodes = 5000
            average_num = 200
            max_reward = 10000
            min_reward = -10000
            aid_num = 2
            plot_key = ['reward', 'reward_nc', 'constraint']
            label_key = ['reward', 'reward_nc', 'Constraint Violation Rate']
            plot_y_lim_dict = {'reward': None,
                               'reward_nc': None,
                               'constraint': None}
            title = 'Blocked Half-Cheetah'
            log_path_dict = {
                "sanity_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2-multi_env": [
                    '../save_model/MEICRL_HCWithPos-v0/sanity_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2-multi_env-Nov-04-2022-17:05-seed_123/',
                ],
                "semi_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-1e-1_advloss-multi_env": [
                    '../save_model/MEICRL_HCWithPos-v0/semi_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-1e-1_advloss-multi_env-Nov-15-2022-20:29-seed_123/',
                ],
                "semi_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-1e-1_closs-multi_env": [
                    '../save_model/MEICRL_HCWithPos-v0/semi_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-1e-1_closs-multi_env-Nov-08-2022-19:50-seed_123/',
                ],
                "semi_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-1e-1-multi_env": [
                    '../save_model/MEICRL_HCWithPos-v0/semi_check-train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-1e-1-multi_env-Nov-07-2022-12:18-seed_123/',
                ],
                "robust_check_0.3_train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-0_robust-3e-1_advloss-multi_env": [
                    '../save_model/MEICRL_HCWithPos-v0/robust_check_0.3_train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-0_robust-3e-1_advloss-multi_env-Nov-18-2022-16:51-seed_123/',
                ],
                "robust_check_0.3_train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-1e-1_robust-3e-1_advloss-multi_env":[
                    '../save_model/MEICRL_HCWithPos-v0/robust_check_0.3_train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-1e-1_robust-3e-1_advloss-multi_env-Nov-18-2022-16:46-seed_123/',
                ],
                "robust_check_0.3_train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-3e-1_robust-3e-1_advloss-multi_env": [
                    '../save_model/MEICRL_HCWithPos-v0/robust_check_0.3_train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-3e-1_robust-3e-1_advloss-multi_env-Nov-18-2022-16:49-seed_123/',
                ],
                "robust_check_0.3_train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_robust-3e-1_advloss-multi_env": [
                    '../save_model/MEICRL_HCWithPos-v0/robust_check_0.3_train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_robust-3e-1_advloss-multi_env-Nov-18-2022-16:49-seed_123/',
                ],
                "robust_check_0.2_train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-0_robust-2e-1_advloss-multi_env": [
                    '../save_model/MEICRL_HCWithPos-v0/robust_check_0.2_train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-0_robust-2e-1_advloss-multi_env-Nov-18-2022-16:51-seed_123/',
                ],
                "robust_check_0.2_train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-1e-1_robust-2e-1_advloss-multi_env": [
                    '../save_model/MEICRL_HCWithPos-v0/robust_check_0.2_train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-1e-1_robust-2e-1_advloss-multi_env-Nov-18-2022-16:46-seed_123/',
                ],
                "robust_check_0.2_train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-3e-1_robust-2e-1_advloss-multi_env": [
                    '../save_model/MEICRL_HCWithPos-v0/robust_check_0.2_train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-3e-1_robust-2e-1_advloss-multi_env-Nov-19-2022-21:04-seed_123/',
                ],
                "robust_check_0.2_train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_robust-2e-1_advloss-multi_env": [
                    '../save_model/MEICRL_HCWithPos-v0/robust_check_0.2_train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_robust-2e-1_advloss-multi_env-Nov-19-2022-21:04-seed_123/',
                ],
                "robust_check_0.1_train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-0_robust-1e-1_advloss-multi_env": [
                    '../save_model/MEICRL_HCWithPos-v0/robust_check_0.1_train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-0_robust-1e-1_advloss-multi_env-Nov-16-2022-17:52-seed_123/',
                ],
                "robust_check_0.1_train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-1e-1_robust-1e-1_advloss-multi_env": [
                    '../save_model/MEICRL_HCWithPos-v0/robust_check_0.1_train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-1e-1_robust-1e-1_advloss-multi_env-Nov-16-2022-17:52-seed_123/',
                ],
                "robust_check_0.1_train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-3e-1_robust-1e-1_advloss-multi_env": [
                    '../save_model/MEICRL_HCWithPos-v0/robust_check_0.1_train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-3e-1_robust-1e-1_advloss-multi_env-Nov-19-2022-21:04-seed_123/',
                ],
                "robust_check_0.1_train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_robust-1e-1_advloss-multi_env": [
                    '../save_model/MEICRL_HCWithPos-v0/robust_check_0.1_train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e-1_robust-1e-1_advloss-multi_env-Nov-19-2022-21:04-seed_123/',
                ],
            }
        elif env_id == 'WGW-v0':
            max_episodes = 1000
            average_num = 100
            max_reward = 1.0
            min_reward = -1.0
            aid_num = 3
            plot_key = ['reward', 'reward_nc', 'constraint']
            label_key = ['reward', 'Feasible Cumulative Rewards', 'Constraint Violation Rate']
            plot_y_lim_dict = {'reward': None,
                               'reward_nc': None,
                               'constraint': None}
            title = 'Gridworld-Setting4'
            log_path_dict = {
                'train_Binary_WGW-v0': [
                    '../save_model/Binary-WallGrid/train_Binary_WGW-v0-May-05-2023-12_14-seed_321/',
                    '../save_model/Binary-WallGrid/train_Binary_WGW-v0-May-04-2023-13_47-seed_123/',
                    '../save_model/Binary-WallGrid/train_Binary_WGW-v0-May-05-2023-13_01-seed_666/',
                ],
                'train_MMICRL_WGW-v0': [
                    # '../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v0_max_nu-1e0-Apr-21-2023-12_12-seed_321/',
                    # '../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v0_max_nu-1e0-Apr-21-2023-12_53-seed_666/',
                    # '../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v0_max_nu-1e0-Mar-13-2023-10_07-seed_123/',
                    '../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v0-May-17-2023-11_07-seed_123/',
                    '../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v0-May-17-2023-13_23-seed_321/',
                    '../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v0-May-17-2023-15_45-seed_666/',
                ],
                'train_MEICRL_WGW-v0': [
                    '../save_model/ICRL-WallGrid/train_ICRL_WGW-v0-Apr-25-2023-19_57-seed_123/',
                    '../save_model/ICRL-WallGrid/train_ICRL_WGW-v0-Apr-26-2023-01_58-seed_321/',
                    '../save_model/ICRL-WallGrid/train_ICRL_WGW-v0-May-03-2023-11_10-seed_666/',
                ],
                'train_MMICRL-LD_WGW-v0': [
                    '../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v0_den-Apr-23-2023-19_40-seed_123/',
                    '../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v0_den-Apr-23-2023-20_33-seed_321/',
                    '../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v0_den-Apr-23-2023-21_31-seed_666/',
                ],
                'train_InfoGAIL-ICRL_WGW-v0': [
                    '../save_model/InfoICRL-WallGrid/train_InfoICRL_WGW-v0-Apr-18-2023-10_01-seed_123/',
                    '../save_model/InfoICRL-WallGrid/train_InfoICRL_WGW-v0-Apr-19-2023-08_07-seed_321/',
                    '../save_model/InfoICRL-WallGrid/train_InfoICRL_WGW-v0-Apr-22-2023-16_37-seed_666/',
                ],
                'train_Binary_WGW-v1': [
                    '../save_model/Binary-WallGrid/train_Binary_WGW-v1-May-04-2023-16_53-seed_123/',
                    '../save_model/Binary-WallGrid/train_Binary_WGW-v1-May-04-2023-18_39-seed_666/',
                    '../save_model/Binary-WallGrid/train_Binary_WGW-v1-May-04-2023-22_23-seed_321/',
                ],
                'train_MMICRL_WGW-v1': [
                    # '../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v1-Apr-14-2023-14_30-seed_123/',
                    # '../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v1-Apr-14-2023-22_26-seed_666/',
                    # '../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v1-Apr-21-2023-15_35-seed_321/'
                    '../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v1-May-23-2023-10_07-seed_321/',
                    '../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v1-May-23-2023-12_02-seed_123/',
                    '../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v1-May-23-2023-14_14-seed_666/',
                ],
                'train_MEICRL_WGW-v1': [
                    '../save_model/ICRL-WallGrid/train_ICRL_WGW-v1-Apr-26-2023-02_56-seed_123/',
                    '../save_model/ICRL-WallGrid/train_ICRL_WGW-v1-Apr-26-2023-12_48-seed_321/',
                    '../save_model/ICRL-WallGrid/train_ICRL_WGW-v1-May-03-2023-11_58-seed_666/',
                ],
                'train_MMICRL-LD_WGW-v1': [
                    '../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v1_den-Apr-23-2023-22_19-seed_123/',
                    '../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v1_den-Apr-24-2023-00_20-seed_321/',
                    '../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v1_den-Apr-24-2023-02_14-seed_666/',
                ],
                'train_InfoGAIL-ICRL_WGW-v1': [
                    '../save_model/InfoICRL-WallGrid/train_InfoICRL_WGW-v1-Apr-18-2023-11_57-seed_123/',
                    '../save_model/InfoICRL-WallGrid/train_InfoICRL_WGW-v1-Apr-22-2023-17_53-seed_321/',
                    '../save_model/InfoICRL-WallGrid/train_InfoICRL_WGW-v1-Apr-23-2023-03_11-seed_666/',
                ],
                'train_Binary_WGW-v2': [
                    '../save_model/Binary-WallGrid/train_Binary_WGW-v2-May-04-2023-14_52-seed_123/',
                    '../save_model/Binary-WallGrid/train_Binary_WGW-v2-May-04-2023-20_23-seed_666/',
                    '../save_model/Binary-WallGrid/train_Binary_WGW-v2-May-05-2023-09_56-seed_321/',
                ],
                'train_MMICRL_WGW-v2': [
                    # '../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v2-Apr-21-2023-21_44-seed_321/',
                    # '../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v2-Apr-21-2023-23_46-seed_666/',
                    # '../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v2-Apr-15-2023-00_15-seed_123/'
                    '../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v2-May-23-2023-16_45-seed_666/',
                    '../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v2-May-23-2023-19_37-seed_321/',
                    '../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v2-May-23-2023-21_35-seed_123/',
                ],
                'train_MEICRL_WGW-v2': [
                    '../save_model/ICRL-WallGrid/train_ICRL_WGW-v2-Apr-26-2023-05_02-seed_123/',
                    '../save_model/ICRL-WallGrid/train_ICRL_WGW-v2-Apr-26-2023-15_37-seed_321/',
                    '../save_model/ICRL-WallGrid/train_ICRL_WGW-v2-May-03-2023-13_40-seed_666/',
                ],
                'train_MMICRL-LD_WGW-v2': [
                    '../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v2_den-Apr-24-2023-04_17-seed_123/',
                    '../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v2_den-Apr-24-2023-06_27-seed_321/',
                    '../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v2_den-Apr-24-2023-08_31-seed_666/',
                ],
                'train_InfoGAIL-ICRL_WGW-v2': [
                    '../save_model/InfoICRL-WallGrid/train_InfoICRL_WGW-v2-Apr-18-2023-14_57-seed_123/',
                    '../save_model/InfoICRL-WallGrid/train_InfoICRL_WGW-v2-Apr-22-2023-20_13-seed_321/',
                    '../save_model/InfoICRL-WallGrid/train_InfoICRL_WGW-v2-Apr-23-2023-05_46-seed_666/',
                ],
                'train_Binary_WGW-v3': [
                    '../save_model/Binary-WallGrid/train_Binary_WGW-v3-Apr-27-2023-10_33-seed_123/',
                    '../save_model/Binary-WallGrid/train_Binary_WGW-v3-Apr-29-2023-00_17-seed_666/',
                    '../save_model/Binary-WallGrid/train_Binary_WGW-v3-May-08-2023-11_49-seed_321/',
                ],
                'train_MMICRL_WGW-v3': [
                    #'../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v4-Apr-23-2023-15_58-seed_123/',
                    #* '../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v4-Apr-24-2023-17_31-seed_321/',
                    # '../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v4-May-16-2023-14_29-seed_123/',
                    # '../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v4-May-16-2023-18_46-seed_666/',
                    #* '../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v4-May-24-2023-14_56-seed_666/',
                    #* '../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v4-May-24-2023-18_29-seed_123/'
                    #'../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v4_clr-5e-3-Apr-22-2023-02_10-seed_123/',
                    #'../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v4-May-09-2023-16_05-seed_666/',
                    #'../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v4_clr-5e-3-Apr-22-2023-05_46-seed_321/'
                    #'../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v4_clr-5e-3-Apr-15-2023-02_12-seed_123/'
                    '../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v4-Aug-05-2023-17_28-seed_666/'
                ],
                'train_MEICRL_WGW-v3': [
                    '../save_model/ICRL-WallGrid/train_ICRL_WGW-v3-Apr-27-2023-14_03-seed_123/',
                    '../save_model/ICRL-WallGrid/train_ICRL_WGW-v3-Apr-28-2023-07_38-seed_321/',
                    '../save_model/ICRL-WallGrid/train_ICRL_WGW-v3-Apr-28-2023-10_58-seed_666/',

                ],
                'train_MMICRL_WGW-v4':[
                    '../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v5-Aug-08-2023-14_50-seed_666/',
                    '../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v5-Aug-08-2023-17_12-seed_123/',
                    '../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v5-Aug-08-2023-20_00-seed_321/',
                ],
                'train_MMICRL-LD_WGW-v3': [
                    '../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v4_den-Apr-25-2023-02_58-seed_666/',
                    '../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v4_den-Apr-25-2023-06_40-seed_321/',
                    '../save_model/MEICRL-WallGrid/train_MEICRL_WGW-v4_den-Apr-25-2023-14_52-seed_123/',
                ],
                'train_InfoGAIL-ICRL_WGW-v3': [
                    '../save_model/InfoICRL-WallGrid/train_InfoICRL_WGW-v4-Apr-18-2023-20_17-seed_123/',
                    '../save_model/InfoICRL-WallGrid/train_InfoICRL_WGW-v4-Apr-19-2023-02_45-seed_321/',
                    '../save_model/InfoICRL-WallGrid/train_InfoICRL_WGW-v4-Apr-22-2023-22_40-seed_666/',
                ],
            }
        else:
            raise ValueError("Unknown env id {0}".format(env_id))

        all_mean_dict_by_cid = {}
        all_std_dict_by_cid = {}
        for aid in range(aid_num):
            all_mean_dict_by_cid.update({aid: {}})
            all_std_dict_by_cid.update({aid: {}})
        for method_name in method_names_labels_dict.keys():
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
                    print(method_name)
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
                                 title=title+ '-z_{0}'.format(aid),
                                 axis_size=axis_size,
                                 img_size=img_size,
                                 )
        for aid in range(aid_num):
            for idx in range(len(plot_key)):
                mean_results_moving_avg_dict = {}
                std_results_moving_avg_dict = {}
                for method_name in method_names_labels_dict.keys():
                    mean_results_moving_avg_dict.update(
                        {method_name: all_mean_dict_by_cid[aid][method_name][plot_key[idx]]})
                    std_results_moving_avg_dict.update(
                        {method_name: all_std_dict_by_cid[aid][method_name][plot_key[idx]]})
                    if (plot_key[idx] == 'reward_nc' or plot_key[idx] == 'constraint') and mode == 'test':
                        print(method_name, plot_key[idx],
                              all_mean_dict_by_cid[aid][method_name][plot_key[idx]][-1],
                              all_std_dict_by_cid[aid][method_name][plot_key[idx]][-1])
                plot_results(mean_results_moving_avg_dict=mean_results_moving_avg_dict,
                             std_results_moving_avg_dict=std_results_moving_avg_dict,
                             label=label_key[idx],
                             method_names=list(method_names_labels_dict.keys()),
                             ylim=plot_y_lim_dict[plot_key[idx]],
                             save_label=os.path.join(env_id, plot_key[idx] + '_c{0}_'.format(aid) + '_'
                                                     + mode + '_' + env_id + '_' + plot_mode),
                             # legend_size=18,
                             legend_dict=method_names_labels_dict,
                             title=title+ '-z_{0}'.format(aid),
                             axis_size=axis_size,
                             img_size=img_size,
                             )


if __name__ == "__main__":
    generate_plots()
