from gym.envs.registration import register

from environments.gazebo_env import GazeboEnv

register(
    id='PlaneBall-v0',
    entry_point='environments.planeball:PlaneBallEnv'
)

register(
    id='ContinuousCartPole-v0',
    entry_point='environments.continuous_cartpole:ContinuousCartPoleEnv',
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id='DiscretePlaneBall-v0',
    entry_point='environments.discrete_planeball:DiscretePlaneBallEnv'
)

