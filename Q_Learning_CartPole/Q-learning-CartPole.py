import gym
import math
from Q_learning_Brain import QLearningTable




env = gym.make('CartPole-v0')

NUM_BUCKETS = [1, 1, 6, 3]
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
STATE_BOUNDS[1] = [-0.5, 0.5]
STATE_BOUNDS[3] = [-math.radians(50), math.radians(50)]


def get_explore_rate(t):
    return max(0.01, min(1, 1.0 - math.log10((t+1)/25)))

def get_learning_rate(t):
    return max(0.1, min(0.5, 1.0 - math.log10((t+1)/25)))


def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)

def update():
    learning_rate = get_learning_rate(0)
    explore_rate = get_explore_rate(0)
    num_streaks = 0
    for episode in range(2000):

        observation = env.reset()

        obv = state_to_bucket(observation)

        for t in range(250):
            env.render()

            action = RL.choose_action(obv, explore_rate)
            observation_, reward, done, info = env.step(action)
            obv_ = state_to_bucket(observation_)
            q_table = RL.learn(obv, action, reward, obv_, learning_rate)
            obv = obv_

            print("\nEpisode = %d" % episode)
            print("t = %d" % t)
            print("Action: %d" % action)
            print("State: %s" % str(obv))
            print("Reward: %f" % reward)
            print("Streaks: %d" % num_streaks)
            print("Explore rate: %f" % explore_rate)
            print("Learning rate: %f" % learning_rate)
            print("Q_table:", q_table)
            print("")

            if done:
                print("Episode %d finished after %f time steps" % (episode, t))
                if (t >= 199):
                    num_streaks += 1
                else:
                    num_streaks = 0
                break
        if num_streaks > 120:
            print("Find the solution in Episode %d . " % episode)
            break

        learning_rate = get_learning_rate(episode)
        explore_rate = get_explore_rate(episode)

    print('game over!')


if __name__ == "__main__":

    RL = QLearningTable(actions=list(range(env.action_space.n)))

 #   env.after(100, update)
#  env.mainloop()

    update()