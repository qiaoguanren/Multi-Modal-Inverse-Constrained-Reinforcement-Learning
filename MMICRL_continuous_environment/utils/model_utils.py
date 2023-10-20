import gym
import torch
import numpy as np
from numpy.linalg import norm
from torch.nn import functional as F


# def get_net_arch(config):
#     """
#     Returns a dictionary with sizes of layers in policy network,
#     value network and cost value network.
#     """
#     try:
#         separate_layers = dict(pi=config.policy_layers,  # Policy Layers
#                                vf=config.reward_vf_layers,  # Value Function Layers
#                                cvf=config.cost_vf_layers)  # Cost Value Function Layers
#     except:
#         print("Could not define layers for policy, value func and " + \
#               "cost_value_function, will attempt to just define " + \
#               "policy and value func")
#         separate_layers = dict(pi=config.policy_layers,  # Policy Layers
#                                vf=config.reward_vf_layers)  # Value Function Layers
#
#     if config.shared_layers is not None:
#         return [*config.shared_layers, separate_layers]
#     else:
#         return [separate_layers]

def get_net_arch(config, log_file):
    """
    Returns a dictionary with sizes of layers in policy network,
    value network and cost value network.
    """
    if 'diversity_vf_layers' in config['PPO'].keys():
        separate_layers = dict(pi=config['PPO']['policy_layers'],  # Policy Layers
                               vf=config['PPO']['reward_vf_layers'],  # Value Function Layers
                               cvf=config['PPO']['cost_vf_layers'],  # Cost Value Function Layers
                               dvf=config['PPO']['diversity_vf_layers'])  # Diversity Value Function Layers
    elif 'cost_vf_layers' in config['PPO'].keys():
        separate_layers = dict(pi=config['PPO']['policy_layers'],  # Policy Layers
                               vf=config['PPO']['reward_vf_layers'],  # Value Function Layers
                               cvf=config['PPO']['cost_vf_layers'])  # Cost Value Function Layers
    else:
        separate_layers = dict(pi=config['PPO']['policy_layers'],  # Policy Layers
                               vf=config['PPO']['reward_vf_layers'])  # Value Function Layers

    print("PPO layers are:", separate_layers, flush=True, file=log_file)
    return [separate_layers]


def handle_model_parameters(model, active_keywords, model_name, log_file, set_require_grad):
    """determine which parameters should be fixed"""
    # exclude some parameters from optimizer
    param_frozen_list = []  # should be changed into torch.nn.ParameterList()
    param_active_list = []  # should be changed into torch.nn.ParameterList()
    fixed_parameters_keys = []
    active_parameters_keys = []
    parameters_info = []

    for k, v in model.named_parameters():
        keep_this = False
        size = torch.numel(v)
        parameters_info.append("{0}:{1}".format(k, size))
        for keyword in active_keywords:
            if keyword in k:
                param_active_list.append(v)
                active_parameters_keys.append(k)
                keep_this = True
                break
        if not keep_this:
            param_frozen_list.append(v)
            if set_require_grad:
                v.requires_grad = False  # fix the parameters https://pytorch.org/docs/master/notes/autograd.html
            fixed_parameters_keys.append(k)

    print('-' * 30 + '{0} Optimizer'.format(model_name) + '-' * 30, file=log_file, flush=True)
    print("Active parameters are: {0}".format(str(active_parameters_keys)), file=log_file, flush=True)
    print("Fixed parameters are: {0}".format(str(fixed_parameters_keys)), file=log_file, flush=True)
    # print(parameters_info, file=log_file, flush=True)
    param_frozen_list = torch.nn.ParameterList(param_frozen_list)
    param_active_list = torch.nn.ParameterList(param_active_list)
    print('-' * 60, file=log_file, flush=True)

    return param_frozen_list, param_active_list


def masked_softmax(x, m=None, axis=-1):
    '''
    Softmax with mask (optional)
    '''
    x = torch.clamp(x, min=-15.0, max=15.0)
    if m is not None:
        m = m.float()
        x = x * m
    e_x = torch.exp(x - torch.max(x, dim=axis, keepdim=True)[0])
    if m is not None:
        e_x = e_x * m
    softmax = e_x / (torch.sum(e_x, dim=axis, keepdim=True) + 1e-6)
    return softmax


def stability_loss(input_data, aggregates, concepts, relevances):
    """Computes Robustness Loss for the Compas data

    Formulated by Alvarez-Melis & Jaakkola (2018)
    [https://papers.nips.cc/paper/8003-towards-robust-interpretability-with-self-explaining-neural-networks.pdf]
    The loss formulation is specific to the data format
    The concept dimension is always 1 for this project by design
    Parameters
    ----------
    input_data   : torch.tensor
                 Input as (batch_size x num_features)
    aggregates   : torch.tensor
                 Aggregates from SENN as (batch_size x num_classes x concept_dim)
    concepts     : torch.tensor
                 Concepts from Conceptizer as (batch_size x num_concepts x concept_dim)
    relevances   : torch.tensor
                 Relevances from Parameterizer as (batch_size x num_concepts x num_classes)

    Returns
    -------
    robustness_loss  : torch.tensor
        Robustness loss as frobenius norm of (batch_size x num_classes x num_features)
    """
    batch_size = input_data.size(0)
    num_classes = aggregates.size(1)

    grad_tensor = torch.ones(batch_size, num_classes).to(input_data.device)
    J_yx = torch.autograd.grad(outputs=aggregates,
                               inputs=input_data,
                               grad_outputs=grad_tensor,
                               create_graph=True,
                               only_inputs=True)[0]
    # bs x num_features -> bs x num_features x num_classes
    J_yx = J_yx.unsqueeze(-1)

    # J_hx = Identity Matrix; h(x) is identity function
    robustness_loss = J_yx - relevances
    robustness_loss = robustness_loss.norm(p='fro', dim=1)
    return robustness_loss


def dirichlet_kl_divergence_loss(alpha, prior):
    """
    KL divergence between two dirichlet distribution
    The mean is alpha/(alpha+beta) and variance is alpha*beta/(alpha+beta)^2*(alpha+beta+1)
    There are multiple ways of modelling a dirichlet:
    1) by Laplace approximation with logistic normal: https://arxiv.org/pdf/1703.01488.pdf
    2) by directly modelling dirichlet parameters: https://arxiv.org/pdf/1901.02739.pdf
    code reference：
    1） https://github.com/sophieburkhardt/dirichlet-vae-topic-models
    2） https://github.com/is0383kk/Dirichlet-VAE
    """
    analytical_kld = torch.lgamma(torch.sum(alpha, dim=1)) - torch.lgamma(torch.sum(prior, dim=1))
    analytical_kld += torch.sum(torch.lgamma(prior), dim=1)
    analytical_kld -= torch.sum(torch.lgamma(alpha), dim=1)
    minus_term = alpha - prior
    # tmp = torch.reshape(torch.digamma(torch.sum(alpha, dim=1)), shape=[alpha.shape[0], 1])
    digamma_term = torch.digamma(alpha) - \
                   torch.reshape(torch.digamma(torch.sum(alpha, dim=1)), shape=[alpha.shape[0], 1])
    test = torch.sum(torch.mul(minus_term, digamma_term), dim=1)
    analytical_kld += test
    # self.analytical_kld = self.mask * self.analytical_kld  # mask paddings
    return analytical_kld


def torch_kron_prod(a, b):
    """
    :param a: matrix1 of size [b, M]
    :param b: matrix2 of size [b, N]
    :return: matrix of size [b, M, N]
    """
    res = torch.einsum('ij,ik->ijk', [a, b])
    res = torch.reshape(res, [-1, np.prod(res.shape[1:])])
    return res


def load_policy_iteration_config(config, env_configs, train_env, seed, log_file):
    pi_parameters = {
        "env": train_env,
        "seed": seed,
        "stopping_threshold": config["iteration"]["stopping_threshold"],
        "max_iter": config["iteration"]["max_iter"],
        "gamma": config["iteration"]["gamma"],
        "n_actions": env_configs['n_actions'],
        "height": env_configs['map_height'],
        "width": env_configs['map_width'],
        "terminal_states": env_configs['terminal_states'],
        "penalty_initial_value": config['iteration']['penalty_initial_value'],
        "penalty_learning_rate": config['iteration']['penalty_learning_rate'],
        "log_file": log_file,

    }
    if config["group"] == "MEICRL":
        pi_parameters.update({"aid": config['CN']['aid']})
        pi_parameters.update({"latent_dim": config['CN']['latent_dim']})
        pi_parameters.update({"penalty_min_value": config['iteration']['nu_min_clamp']})
        pi_parameters.update({"penalty_max_value": config['iteration']['nu_max_clamp']})
    return pi_parameters


def load_ppo_config(config, train_env, seed, log_file):
    ppo_parameters = {
        "policy": config['PPO']['policy_name'],
        "env": train_env,
        "learning_rate": config['PPO']['learning_rate'],
        "n_steps": config['PPO']['n_steps'],
        "batch_size": config['PPO']['batch_size'],
        "n_epochs": config['PPO']['n_epochs'],
        "clip_range": config['PPO']['clip_range'],
        "ent_coef": config['PPO']['ent_coef'],
        "max_grad_norm": config['PPO']['max_grad_norm'],
        "use_sde": config['PPO']['use_sde'],
        "sde_sample_freq": config['PPO']['sde_sample_freq'],
        "target_kl": config['PPO']['target_kl'],
        "verbose": config['verbose'],
        "seed": seed,
        "device": config['device'],
        "policy_kwargs": dict(net_arch=get_net_arch(config, log_file))
    }
    if config["group"] == "PPO" or config["group"] == "GAIL":
        ppo_parameters.update({
            "gamma": config['PPO']['reward_gamma'],
            "gae_lambda": config['PPO']['reward_gae_lambda'],
            "vf_coef": config['PPO']['reward_vf_coef'],
        })
    elif config['group'] == "PPO-Lag" or config['group'] == "Binary" \
            or config['group'] == "ICRL" or config['group'] == "VICRL" \
            or config['group'] == "MEICRL" or config['group'] == "InfoICRL":
        # elif config['group'] == "PPO-Lag":
        ppo_parameters.update({
            "reward_gamma": config['PPO']['reward_gamma'],
            "reward_gae_lambda": config['PPO']['reward_gae_lambda'],
            "cost_gamma": config['PPO']['cost_gamma'],
            "cost_gae_lambda": config['PPO']['cost_gae_lambda'],
            "clip_range_reward_vf": config['PPO']['clip_range_reward_vf'],
            "clip_range_cost_vf": config['PPO']['clip_range_cost_vf'],
            "reward_vf_coef": config['PPO']['reward_vf_coef'],
            "cost_vf_coef": config['PPO']['cost_vf_coef'],
            "penalty_initial_value": config['PPO']['penalty_initial_value'],
            "penalty_learning_rate": config['PPO']['penalty_learning_rate'],
            "budget": config['PPO']['budget'],
            "pid_kwargs": dict(alpha=config['PPO']['budget'],
                               penalty_init=config['PPO']['penalty_initial_value'],
                               Kp=config['PPO']['proportional_control_coeff'],
                               Ki=config['PPO']['integral_control_coeff'],
                               Kd=config['PPO']['derivative_control_coeff'],
                               pid_delay=config['PPO']['pid_delay'],
                               delta_p_ema_alpha=config['PPO']['proportional_cost_ema_alpha'],
                               delta_d_ema_alpha=config['PPO']['derivative_cost_ema_alpha'], ),
        })
        if config['group'] == "MEICRL" or config['group'] == "InfoICRL" or config['group'] == "ICRL" or config['group'] == "Binary":
            ppo_parameters.update({"latent_dim": config['CN']['latent_dim']})
            ppo_parameters["policy_kwargs"].update({"latent_dim": config['CN']['latent_dim']})
            ppo_parameters.update({"aid": config['CN']['aid']})
            ppo_parameters.update({"n_probings": config['CN']['n_probings']})
            ppo_parameters.update({"contrastive_weight": config['CN']['contrastive_weight']})
            ppo_parameters.update({"log_cost": config['PPO']['log_cost']})
            ppo_parameters.update({"contrastive_augment_type": config['PPO']['contrastive_augment_type']})
            if config['PPO']['contrastive_augment_type'] == 'calculate advantages':
                ppo_parameters.update({"diversity_vf_coef": config['PPO']['diversity_vf_coef']})
                ppo_parameters.update({"diversity_gamma": config['PPO']['diversity_gamma']})
                ppo_parameters.update({"diversity_gae_lambda": config['PPO']['diversity_gae_lambda']})
                ppo_parameters.update({"clip_range_cost_vf": config['PPO']['clip_range_cost_vf']})
    else:
        raise ValueError("Unknown Group {0}".format(config['group']))

    return ppo_parameters


def build_code(code_axis, code_dim, num_envs):
    code = np.zeros((num_envs, code_dim), dtype=np.float32)
    for env_idx in range(num_envs):
        code[env_idx, code_axis[env_idx]] = 1
    return code


def update_code(code_axis, code_dim):
    code_axis = (code_axis + 1) % code_dim
    return code_axis


def diversity_return_function(observations, actions, pos_latent_signals, neg_latent_signals):
    pos_sample_size = pos_latent_signals.shape[1]
    neg_sample_size = neg_latent_signals.shape[1] * neg_latent_signals.shape[2]
    observations = to_np(observations)
    actions = to_np(actions)
    pos_latent_signals = to_np(pos_latent_signals)
    neg_latent_signals = to_np(neg_latent_signals)
    obs_act = np.concatenate([observations, actions], axis=1)
    obs_act_repeat = np.expand_dims(obs_act, axis=1).repeat(repeats=neg_sample_size, axis=1)
    act_obs_dim = obs_act_repeat.shape[-1]
    c_log_probs = []
    for pos_id in range(pos_sample_size):
        pos_similarity = cosine_similarity(pos_latent_signals[:, pos_id, :], obs_act)
        neg_similarity_all = cosine_similarity(neg_latent_signals.reshape(-1, act_obs_dim),
                                               obs_act_repeat.reshape(-1, act_obs_dim)).reshape([-1, neg_sample_size])
        neg_similarity_sum_exp = np.exp(neg_similarity_all).sum(axis=1)
        c_prob_pid = np.exp(pos_similarity) / (neg_similarity_sum_exp + np.exp(pos_similarity))
        c_log_probs.append(c_prob_pid)
    contrastive_loss = np.stack(c_log_probs, axis=1).mean(axis=1)
    contrastive_loss = -(np.log(contrastive_loss) - np.log(1/(neg_sample_size+1)))
    return contrastive_loss


def cosine_similarity(A, B):
    cosine = []
    for i in range(A.shape[0]):
        cosine.append(np.dot(A[i, :], B[i, :]) / (norm(A[i, :]) * norm(B[i, :])))
    return np.asarray(cosine)


def to_np(x):
    if isinstance(x, np.ndarray):
        return x
    return x.data.cpu().numpy()
