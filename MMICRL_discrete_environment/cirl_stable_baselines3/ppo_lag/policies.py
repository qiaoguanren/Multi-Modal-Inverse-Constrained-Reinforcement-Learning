# This file is here just to define the TwoCriticsPolicy for PPO-Lagrangian
from cirl_stable_baselines3.common.policies import (ActorTwoCriticsPolicy,
                                                    ActorTwoCriticsCnnPolicy,
                                                    register_policy,
                                                    ActorThreeCriticsPolicy)

TwoCriticsMlpPolicy = ActorTwoCriticsPolicy
ThreeCriticsMlpPolicy = ActorThreeCriticsPolicy
TwoCriticsCnnPolicy = ActorTwoCriticsCnnPolicy

register_policy("TwoCriticsMlpPolicy", ActorTwoCriticsPolicy)
register_policy("ThreeCriticsMlpPolicy", ActorThreeCriticsPolicy)
register_policy("TwoCriticsCnnPolicy", ActorTwoCriticsCnnPolicy)
