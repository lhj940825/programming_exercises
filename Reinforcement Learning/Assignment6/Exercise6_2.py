import math
import numpy as np
import matplotlib.pyplot as plt
from CartPole import CartPoleEnv


def main():

    env = CartPoleEnv()
    numEpisodes = 1000
    final_rewards = []
    for _ in range(numEpisodes):
        env.reset()
        final_reward = (-1)*numEpisodes
        t = 0
        while True:
            action = env.action_space.sample()
            env.getNextState(action)
            reward = env.getReward()
            if env.isTerminal:
                final_reward += (t+1)
                final_rewards.append(final_reward)
                print("Episode finished after {} timesteps".format(t+1))
                break
            t += 1
    env.close()
    fig, ax = plt.subplots()
    ax.plot(final_rewards)
    ax.set(xlabel='episodes', ylabel='final reward')
    ax.grid()
    plt.show()

if __name__ == "__main__":
    main()