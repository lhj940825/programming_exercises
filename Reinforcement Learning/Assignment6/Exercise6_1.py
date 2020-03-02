import math
import numpy as np
import matplotlib.pyplot as plt
from CartPole import CartPoleEnv

def plot_final_reward_graph(pos_history, vel_history, angle_history, ang_vel_history):
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    fig.suptitle("Exercise 6.1")
    ax1 = fig.add_subplot(2,2,1)
    ax1.title.set_text("position")
    ax1.plot(pos_history)
    ax1.set(ylabel="meter", xlabel="step")

    ax2 = fig.add_subplot(2,2,2)
    ax2.title.set_text("velocity")
    ax2.plot(vel_history)
    ax2.set(ylabel="meter/sec", xlabel="step")

    ax3 = fig.add_subplot(2,2,3)
    ax3.title.set_text("angle")
    ax3.plot(angle_history)
    ax3.set(ylabel="rad", xlabel="step")

    ax4 = fig.add_subplot(2,2,4)
    ax4.title.set_text("angular velocity")
    ax4.plot(ang_vel_history)
    ax4.set(ylabel="rad/sec", xlabel="step")


    plt.show()


def main():
    # initial value: position = -1m, velocity = 0.25m/s, angle = 0.3rad, angular velocity = -0.7 rad/s
    pos = -1
    vel = 0.25
    angle = 17.1887* 2 * math.pi / 360
    ang_vel = -40.107* 2 * math.pi / 360

    pos_history =[]
    vel_history =[]
    angle_history =[]
    ang_vel_history =[]
    numEpisodes = 100

    env = CartPoleEnv()
    for i in range(int(1/env.timeStep)):
        pos, vel, angle, ang_vel = env.state_tracking(pos, vel, angle, ang_vel)
        pos_history.append(pos)
        vel_history.append(vel)
        angle_history.append(angle)
        ang_vel_history.append(ang_vel)
        
    for _ in range(numEpisodes):
        t = 0
        env.reset() #For simulation of the cart-pole system
        while True:
            env.render()
            action = env.action_space.sample()
            env.getNextState(action)
            if env.isTerminal:
                print("Episode finished after {} timesteps".format(t+1))
                break
            t += 1


    plot_final_reward_graph(pos_history,vel_history, angle_history, ang_vel_history)


if __name__ == "__main__":
    main()
