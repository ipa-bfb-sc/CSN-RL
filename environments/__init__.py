from gym.envs.registration import register

#from environments.gazebo_env import GazeboEnv

register(
    id='ContinuousCartPole-v0',
    entry_point='environments.CartPole.continuous_cartpole:ContinuousCartPoleEnv',
    max_episode_steps=200,
    reward_threshold=195.0,
)


register(
    id='PlaneBall-v0',
    entry_point='environments.PlaneBall.planeball:PlaneBallEnv',
    max_episode_steps=999,
    reward_threshold=9000.0,
)

register(
    id='DiscretePlaneBall-v0',
    entry_point='environments.PlaneBall.discrete_planeball:DiscretePlaneBallEnv',
    max_episode_steps=999,
    reward_threshold=9000.0,
)


# Turtlebot envs
register(
    id='GazeboCircuit2TurtlebotLidar-v1',
    entry_point='environments.CirTurtleBot.gazebo_circuit2_turtlebot_lidar:GazeboCircuit2TurtlebotLidarEnv',

reward_threshold=2000.0,   
max_episode_steps=300,

 # More arguments here
)

register(
    id='GazeboCircuit2TurtlebotLidarNn-v1',
    entry_point='environments.CirTurtleBot.gazebo_circuit2_turtlebot_lidar_nn:GazeboCircuit2TurtlebotLidarNnEnv',
    # More arguments here
reward_threshold=2000.0,   
max_episode_steps=300,

)

#Pendulum
register(
    id='pendulum-v1',
    entry_point='environments.pendulum:PendulumEnv',
    # More arguments here
max_episode_steps=200,

)
