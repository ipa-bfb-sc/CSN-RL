from gym.envs.registration import register

register(
    id='PlaneBall-v0',
    entry_point='environments.planeball:PlaneBallEnv'
)
