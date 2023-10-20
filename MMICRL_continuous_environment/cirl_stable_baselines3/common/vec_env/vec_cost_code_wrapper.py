import pickle

from cirl_stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
import numpy as np
# =============================================================================
# Cost Wrapper
# =============================================================================
from utils.model_utils import update_code, build_code


class VecCostCodeWrapper(VecEnvWrapper):
    def __init__(self, venv, latent_dim, max_seq_len=None, cost_info_str='cost', latent_info_str='latent_posterior'):
        super().__init__(venv)
        self.cost_info_str = cost_info_str
        self.latent_info_str = latent_info_str
        self.latent_dim = latent_dim
        self.code_axis = [0 for i in range(self.num_envs)]
        self.action_seqs = [[] for i in range(self.num_envs)]
        self.obs_seqs = [[] for i in range(self.num_envs)]
        self.max_seq_len = max_seq_len

    def step_async(self, actions: np.ndarray):
        self.actions = actions
        for i in range(self.num_envs):
            if len(self.action_seqs[i]) >= self.max_seq_len:
                self.action_seqs[i].pop(0)
            self.action_seqs[i].append(self.actions[i])
        self.venv.step_async(self.actions)

    def step_async_with_code(self, actions: np.ndarray, codes: np.ndarray):
        self.actions = actions
        self.codes = codes
        for i in range(self.num_envs):
            if len(self.action_seqs[i]) >= self.max_seq_len:
                self.action_seqs[i].pop(0)
            self.action_seqs[i].append(self.actions[i])
        self.venv.games_by_aids = self.games_by_aids
        self.venv.step_async_with_code(self.actions, self.codes)

    def __getstate__(self):
        """
        Gets state for pickling.

        Excludes self.venv, as in general VecEnv's may not be pickleable."""
        state = self.__dict__.copy()
        # these attributes are not pickleable
        del state["venv"]
        del state["class_attributes"]
        # these attributes depend on the above and so we would prefer not to pickle
        del state["cost_function"]
        return state

    def __setstate__(self, state):
        """
        Restores pickled state.

        User must call set_venv() after unpickling before using.

        :param state:"""
        self.__dict__.update(state)
        assert "venv" not in state
        self.venv = None

    def set_venv(self, venv):
        """
        Sets the vector environment to wrap to venv.

        Also sets attributes derived from this such as `num_env`.

        :param venv:
        """
        if self.venv is not None:
            raise ValueError("Trying to set venv of already initialized VecNormalize wrapper.")
        VecEnvWrapper.__init__(self, venv)

    def step_wait(self):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)

        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs, rews, news, infos = self.venv.step_wait()
        if infos is None:
            infos = {}
        # Costs and latent codes depends on previous observation and current actions
        cost_signals = self.cost_function(self.previous_obs.copy(),
                                          self.actions.copy(),
                                          self.codes.copy())  # [batch size]
        latent_signals = self.latent_function(self.previous_obs.copy(),
                                              self.actions.copy(),
                                              self.codes.copy())  # [batch size]
        # if 'history' in self.latent_info_str:
        #     obs_seqs = np.stack([np.asarray(item) for item in self.obs_seqs], axis=0)
        #     action_seqs = np.stack([np.asarray(item) for item in self.action_seqs], axis=0)
        #     code_posterior = None
        # else:
        #     code_posterior = None
        for env_idx in range(self.num_envs):
            if news[env_idx]:  # the game is done
                # remove the code update
                # self.code_axis[env_idx] = update_code(code_dim=self.latent_dim, code_axis=self.code_axis[env_idx])
                # clean up recordings
                self.obs_seqs[env_idx] = []
                self.action_seqs[env_idx] = []
        # self.codes = build_code(code_axis=self.code_axis, code_dim=self.latent_dim, num_envs=self.num_envs)
        for i in range(len(infos)):
            infos[i][self.cost_info_str] = cost_signals[i]
            # tmp = np.argmax(self.code[i])
            infos[i][self.latent_info_str] = latent_signals[np.argmax(self.codes[i])]
            infos[i]['new_code'] = self.codes[i]  # so that the code will not be update
        # record observations
        self.previous_obs = obs.copy()
        for i in range(self.num_envs):
            if len(self.obs_seqs[i]) >= self.max_seq_len:
                self.obs_seqs[i].pop(0)
            self.obs_seqs[i].append(obs.copy()[i])
        return obs, rews, news, infos

    def set_cost_function(self, cost_function):
        self.cost_function = cost_function

    def set_latent_function(self, latent_function):
        self.latent_function = latent_function

    def reset(self):
        """
        Reset all environments
        """
        obs = self.venv.reset()
        self.previous_obs = obs
        self.action_seqs = [[] for i in range(self.num_envs)]
        self.obs_seqs = [[obs[i]] for i in range(self.num_envs)]
        self.codes = build_code(code_axis=self.code_axis, code_dim=self.latent_dim, num_envs=self.num_envs)
        return obs
    def reset_with_values(self, info_dicts):
        """
        Reset all environments
        """
        obs = self.venv.reset_with_values(info_dicts)
        self.previous_obs = obs
        self.action_seqs = [[] for i in range(self.num_envs)]
        self.obs_seqs = [[obs[i]] for i in range(self.num_envs)]
        self.codes = build_code(code_axis=self.code_axis, code_dim=self.latent_dim, num_envs=self.num_envs)
        return obs
    @staticmethod
    def load(load_path: str, venv: VecEnv):
        """
        Loads a saved VecCostWrapper object.

        :param load_path: the path to load from.
        :param venv: the VecEnv to wrap.
        :return:
        """
        with open(load_path, "rb") as file_handler:
            vec_cost_wrapper = pickle.load(file_handler)
        vec_cost_wrapper.set_venv(venv)
        return vec_cost_wrapper

    def save(self, save_path: str) -> None:
        """
        Save current VecCostWrapper object with
        all running statistics and settings (e.g. clip_obs)

        :param save_path: The path to save to
        """
        with open(save_path, "wb") as file_handler:
            pickle.dump(self, file_handler)

    def get_image(self):
        print("abc")
        return None
