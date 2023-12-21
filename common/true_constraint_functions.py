from functools import partial
import numpy as np


def get_true_constraint_function(env_id, env_configs={}, agent_id=0, c_id=None, games_by_aids={}):
    """Returns the cost function correpsonding to provided env)"""
    if env_id in ["HCWithPosTest-v0",
                  "AntWallTest-v0",
                  "HCWithPos-v0",
                  "AntWall-v0",
                  ]:
        if c_id is None:
            # games_by_cids = {0: 1, 2: 1, 4: 1, 6: 1, 8: 1,
            #                  1: 0, 3: 0, 5: 0, 7: 0, 9: 0,
            #                  }
            games_by_cids = env_configs['games_by_cids']
            vote = [0 for i in range(len(games_by_aids.keys()))]
            for game_index in games_by_aids[agent_id]:
                cid = games_by_cids[game_index]
                vote[cid] += 1
            c_id = np.argmax(np.asarray(vote))
        if c_id == 0:
            return partial(wall_behind, -3)
        elif c_id == 1:
            return partial(wall_infront, 3)
        else:
            raise ValueError("Unknown cid {0}.".format(c_id))
    elif env_id in ["SwimmerWithPos-v0",
                    "SwimmerWithPosTest-v0"
                    ]:
        if c_id is None:
            games_by_cids = env_configs['games_by_cids']
            vote = [0 for i in range(len(games_by_aids.keys()))]
            for game_index in games_by_aids[agent_id]:
                cid = games_by_cids[game_index]
                vote[cid] += 1
            c_id = np.argmax(np.asarray(vote))
        if c_id == 0:
            return partial(wall_behind, -0.01)  # -0.1
        elif c_id == 1:
            return partial(wall_infront, 0.01)
        else:
            raise ValueError("Unknown cid {0}.".format(c_id))
    elif env_id in ["WalkerWithPos-v0",
                    "WalkerWithPosTest-v0", ]:
        if c_id is None:
                games_by_cids = env_configs['games_by_cids']
                vote = [0 for i in range(len(games_by_aids.keys()))]
                for game_index in games_by_aids[agent_id]:
                    cid = games_by_cids[game_index]
                    vote[cid] += 1
                c_id = np.argmax(np.asarray(vote))
        if c_id == 0:
            return partial(wall_behind, -0.1)
        elif c_id == 1:
            return partial(wall_infront, 0.1)
        else:
            raise ValueError("Unknown cid {0}.".format(c_id))
    elif env_id in ["WGW-v0"]:
        if c_id is None:
            games_by_cids = env_configs['games_by_cids']
            vote = [0 for i in range(len(games_by_aids.keys()))]
            for game_index in games_by_aids[agent_id]:
                cid = games_by_cids[game_index]
                vote[cid] += 1
            c_id = np.argmax(np.asarray(vote))
        unsafe_states = env_configs['unsafe_states']
        return partial(wall_in, unsafe_states[c_id])

    elif env_id in ["Circle-v0", ]:
        return null_cost
    else:
        print("Cost function for %s is not implemented yet. Returning null cost function" % env_id)
        return null_cost

# ============================================================================
# General cost functions
# ============================================================================

def wall_behind(pos, obs, acs):
    return (obs[..., 0] < pos)


def wall_infront(pos, obs, acs):
    return (obs[..., 0] > pos)


def wall_in(unsafe_states, obs, acs):
    return (obs in unsafe_states)


def wall_behind_and_infront(pos_back, pos_front, obs, acs):
    return (obs[..., 0] <= pos_back).astype(np.float32) + (obs[..., 0] >= pos_front).astype(np.float32)


def null_cost(x, *args):
    # Zero cost everywhere
    return 0


def torque_constraint(threshold, obs, acs):
    return np.any(np.abs(acs) > threshold, axis=-1)