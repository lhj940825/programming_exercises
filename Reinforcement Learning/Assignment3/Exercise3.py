"""
Robot Learning Exercise 3
Member 1
Name: Hojun Lim
Mat. No: 3279159

Member 2
Name: Kajaree Das
Mat. No: 3210311

"""

from itertools import product

import numpy as np
import sys
import matplotlib.pyplot as plt


class GridWorld(object):
    # define a 4 different action[up, down, right, left] that agent can take
    class Action:
        up = np.array([-1, 0])
        down = np.array([1, 0])
        right = np.array([0, 1])
        left = np.array([0, -1])

    def __init__(self):
        # define the grid world as [9x9] matrix as the question requires
        # G: goal, reward +100p and the episode ends
        # *: reward +5p, episode continues
        # X: reward -20p, stay in the previous cell
        # outside of world: reward -5p and stay in the previous cell
        #
        self.actions = [self.Action.up, self.Action.down, self.Action.right, self.Action.left]
        self.is_State_final = False
        self.map = np.array([['*', '*', '*', ' ', '*', '*', ' ', '*', '*'],
                             [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', '*'],
                             [' ', 'X', 'X', 'X', 'X', 'X', 'X', ' ', ' '],
                             [' ', ' ', ' ', '*', '*', '*', ' ', 'X', ' '],
                             [' ', ' ', 'X', 'X', 'X', 'X', 'X', ' ', ' '],
                             [' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X', ' '],
                             [' ', 'X', 'X', 'X', 'X', 'X', 'X', 'G', ' '],
                             [' ', '*', '*', '*', 'X', '*', '*', ' ', 'X'],
                             [' ', '*', '*', '*', ' ', '*', '*', ' ', 'X']])
        self.rowSize = 9
        self.colSize = 9
        self.mapSize = self.rowSize * self.colSize
    # input: state(array containing coordinate) and action,
    # return: coordinate of the next state and acquired reward
    def take_action(self, state, action):

        if self.map[state[0], state[1]] == 'G':
            return state, 0.

        new_state = state + action
        # if we try to move out of the grid world, reward -5 p
        if (not new_state[0] in range(0, 9) or
                not new_state[1] in range(0, 9)):
            return state, -5.

        if self.map[tuple(new_state)] == ' ':
            state = new_state
            return state, -1.
        elif self.map[tuple(new_state)] == 'X':
            return state, -20.
        elif self.map[tuple(new_state)] == '*':
            state = new_state
            return state, 5.
        elif self.map[tuple(new_state)] == 'G':
            state = new_state
            self.final_state = True
            return state, 100.
        else:
            raise Exception('Wrong Input Type')

# GridWorld where state can move to 8 neighboring cell(dirrection)
class GridWorld8D(object):
    # define a 4 different action[up, down, right, left] that agent can take
    class Action:
        up = np.array([-1, 0])
        down = np.array([1, 0])
        right = np.array([0, 1])
        left = np.array([0, -1])
        upR = np.array([-1,1])
        upL = np.array([-1,-1])
        doR = np.array([1,1])
        doL = np.array([1,-1])


    def __init__(self):
        # define the grid world as [9x9] matrix as the question requires
        # G: goal, reward +100p and the episode ends
        # *: reward +5p, episode continues
        # X: reward -20p, stay in the previous cell
        # outside of world: reward -5p and stay in the previous cell
        #
        self.actions = [self.Action.up, self.Action.down, self.Action.right, self.Action.left, self.Action.upR, self.Action.upL, self.Action.doR, self.Action.doL]
        self.is_State_final = False
        self.map = np.array([['*', '*', '*', ' ', '*', '*', ' ', '*', '*'],
                             [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', '*'],
                             [' ', 'X', 'X', 'X', 'X', 'X', 'X', ' ', ' '],
                             [' ', ' ', ' ', '*', '*', '*', ' ', 'X', ' '],
                             [' ', ' ', 'X', 'X', 'X', 'X', 'X', ' ', ' '],
                             [' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X', ' '],
                             [' ', 'X', 'X', 'X', 'X', 'X', 'X', 'G', ' '],
                             [' ', '*', '*', '*', 'X', '*', '*', ' ', 'X'],
                             [' ', '*', '*', '*', ' ', '*', '*', ' ', 'X']])
        self.rowSize = 9
        self.colSize = 9
        self.mapSize = self.rowSize * self.colSize
    # input: state(array containing coordinate) and action,
    # return: coordinate of the next state and acquired reward
    def take_action(self, state, action):

        if self.map[state[0], state[1]] == 'G':
            return state, 0.

        new_state = state + action
        # if we try to move out of the grid world, reward -5 p
        if (not new_state[0] in range(0, 9) or
                not new_state[1] in range(0, 9)):
            return state, -5.

        if self.map[tuple(new_state)] == ' ':
            state = new_state
            return state, -1.
        elif self.map[tuple(new_state)] == 'X':
            return state, -20.
        elif self.map[tuple(new_state)] == '*':
            state = new_state
            return state, 5.
        elif self.map[tuple(new_state)] == 'G':
            state = new_state
            self.final_state = True
            return state, 100.
        else:
            raise Exception('Wrong Input Type')



# GridWorld where state can move to 8 neighboring cell(dirrection)
class StochasticGridWorld(object):
    # define a 4 different action[up, down, right, left] that agent can take
    class Action:
        up = np.array([-1, 0])
        down = np.array([1, 0])
        right = np.array([0, 1])
        left = np.array([0, -1])
        upR = np.array([-1,1])
        upL = np.array([-1,-1])
        doR = np.array([1,1])
        doL = np.array([1,-1])


    def __init__(self):
        # define the grid world as [9x9] matrix as the question requires
        # G: goal, reward +100p and the episode ends
        # *: reward +5p, episode continues
        # X: reward -20p, stay in the previous cell
        # outside of world: reward -5p and stay in the previous cell
        #

        #TODO I made a list of actions follwing clockwise order
        # list of actions in clock wise order
        self.actions = [self.Action.up, self.Action.upR, self.Action.right, self.Action.doR, self.Action.down, self.Action.doL, self.Action.left, self.Action.upL]
        self.is_State_final = False
        self.map = np.array([['*', '*', '*', ' ', '*', '*', ' ', '*', '*'],
                             [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', '*'],
                             [' ', 'X', 'X', 'X', 'X', 'X', 'X', ' ', ' '],
                             [' ', ' ', ' ', '*', '*', '*', ' ', 'X', ' '],
                             [' ', ' ', 'X', 'X', 'X', 'X', 'X', ' ', ' '],
                             [' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X', ' '],
                             [' ', 'X', 'X', 'X', 'X', 'X', 'X', 'G', ' '],
                             [' ', '*', '*', '*', 'X', '*', '*', ' ', 'X'],
                             [' ', '*', '*', '*', ' ', '*', '*', ' ', 'X']])
        self.rowSize = 9
        self.colSize = 9
        self.mapSize = self.rowSize * self.colSize
    # input: state(array containing coordinate) and action,
    # return: coordinate of the next state and acquired reward
    def take_action(self, state, action):

        if self.map[state[0], state[1]] == 'G':
            return state, 0.

        new_state = state + action
        # if we try to move out of the grid world, reward -5 p
        if (not new_state[0] in range(0, 9) or
                not new_state[1] in range(0, 9)):
            return state, -5.

        if self.map[tuple(new_state)] == ' ':
            state = new_state
            return state, -1.
        elif self.map[tuple(new_state)] == 'X':
            return state, -20.
        elif self.map[tuple(new_state)] == '*':
            state = new_state
            return state, 5.
        elif self.map[tuple(new_state)] == 'G':
            state = new_state
            self.final_state = True
            return state, 100.
        else:
            raise Exception('Wrong Input Type')


# EXERCISE 3.1
def iter_policy_Evaluation(act_prob, discnt_V, tolerance, policy=None, isPolicy=None):
    gridWorld = GridWorld()

    # initialize variables for storing estimated value map
    # v_current = V(k), v_update = V(K+1)
    v_current = np.zeros([9, 9], dtype=float)
    v_update = np.zeros([9, 9], dtype=float)

    delta = sys.float_info.max

    if isPolicy == None:

        while (delta > tolerance):
            # initialize difference between v_current and v_next as 0 in every sweep
            delta = 0.

            # State = [x][y]
            # x index
            for x in range(9):
                # y index
                for y in range(9):
                    v_current[x][y] = v_update[x][y]

                    # when current state is 'X', skip this iteration
                    if gridWorld.map[x][y] == 'X':
                        continue

                    current_state = np.array([x, y])
                    act_Results = [gridWorld.take_action(current_state, action=action) for action in gridWorld.actions]
                    # update the estimated value in gridWorld[x,y]
                    v_update[x][y] = np.sum(
                        np.multiply([act_prob], [reward + discnt_V * v_current[update_state[0], update_state[1]] \
                                                 for [update_state, reward] in act_Results]))
                    # if x == 0 and y ==0:
                    #     print([act_prob], [reward + discnt_V * v_current[update_state[0], update_state[1]] \
                    #                    for [update_state, reward] in act_Results])
                    delta = max(delta, np.abs(v_current[x][y] - v_update[x][y]))

        # print(v_update)
        # Plot expected value of all cell

        return v_update

    # when the policy is given
    else:

        while (delta > tolerance):
            # initialize difference between v_current and v_next as 0 in every sweep
            delta = 0.

            # State = [x][y]
            # x index
            for x in range(9):
                # y index
                for y in range(9):
                    v_current[x][y] = v_update[x][y]

                    # when current state is 'X', skip this iteration
                    if gridWorld.map[x][y] == 'X':
                        continue

                    current_state = np.array([x, y])
                    act_Results = [gridWorld.take_action(current_state, action=action) for action in gridWorld.actions]

                    # update the estimated value in gridWorld[x,y]
                    v_update[x][y] = np.sum(
                        np.multiply(policy[x * 9 + y], [reward + discnt_V * v_current[update_state[0], update_state[1]] \
                                                        for [update_state, reward] in act_Results]))
                    delta = max(delta, np.abs(v_current[x][y] - v_update[x][y]))

    # Plot expected value of all cell

        return v_update


# Exercise 3.2
def policy_Iteration(act_prob, discnt_V, tolerance):
    # size of gridWorld
    rowSize = 9
    colSize = 9
    mapSize = rowSize * colSize

    # up, down, right, left
    num_action = 4

    # initialize the policies according to random distribution
    policy = np.array([act_prob] * mapSize)

    i = 0
    while True:

        # Evaluate the current policy
        V = iter_policy_Evaluation(discnt_V=discnt_V, tolerance=tolerance, act_prob=None, policy=policy, isPolicy=True)

        i = i + 1
        # will be set to false if we update the policy
        policy_stable = True

        for x in range(9):
            for y in range(9):

                # when current state is 'X' or 'G', skip this iteration
                if GridWorld().map[x][y] == 'X' or GridWorld().map[x][y] == 'G':
                    continue

                # action that currently in policy, policy[x*10+y]: in order to access to the respective policy by state,
                chosen_act = np.argmax(policy[x * 9 + y])

                current_state = np.array([x, y])
                act_Results = [GridWorld().take_action(current_state, action=action) for action in GridWorld().actions]

                # lists of estimated action values
                # probability = [1/num_action]*num_action = [0.25,0.25,0.25,0.25]
                act_values = np.multiply([1/num_action]*num_action, [reward + discnt_V * V[update_state[0], update_state[1]] \
                                                       for [update_state, reward] in act_Results])


                best_act = np.argmax(act_values)
                #print(chosen_act, best_act, policy[x*9+y],act_values)
               # print(chosen_act, best_act, policy[x*9+y], act_values)

                if chosen_act != best_act:
                    policy_stable = False

                policy[x * 9 + y] = np.eye(num_action)[best_act]

        if policy_stable:
            break

    plot_policy(value_function=V, title="Exercise 3.2 Policy Iteration Algorithm")

    
    plot_value(V, "Exercise 3.2 optimal value of all cells")
    return policy, V


# Exercise 3.3
def Value_Iteration(act_prob, discnt_V, tolerance):
    """
    **Exercise 3.2 and 3.3 Comparison**
    Discussion:Compare to policy iteration algorithm of 3.2, we can clearly see that value iteration algorithm of 3.3 converge much earlyier than the algorithmn of 3.2 \
    in terms of calculated optimal values. Moreover, as it is allowed to move in diagonal direction in 3.3, the value iteration algorithm choose a diagonal action\
    when it can reach to the higher value states or the goal quicker, instead of traveling around(indirect way) like the optimal policy of 3.2 does.


    :return:
    """
    gridWorld = GridWorld8D()

    # initialize variables for storing estimated value map
    # v_current = V(k), v_update = V(K+1)
    v_current = np.zeros([9, 9], dtype=float)
    v_update = np.zeros([9, 9], dtype=float)

    #up, down, right, left, upright, upleft, downright, downleft
    num_action = 8

    delta = sys.float_info.max

    # initialize as -1 which denotes block:X, we do not take action in there, action index: [0:3]
    opt_policy=np.zeros([gridWorld.rowSize, gridWorld.colSize])-1


    while (delta > tolerance):
        # initialize difference between v_current and v_next as 0 in every sweep
        delta = 0.

        # State = [x][y]
        # x index
        for x in range(9):
            # y index
            for y in range(9):
                v_current[x][y] = v_update[x][y]

                # when current state is 'X' or 'G', skip this iteration
                if GridWorld().map[x][y] == 'X' or GridWorld().map[x][y] == 'G':
                    continue

                current_state = np.array([x, y])
                act_Results = [gridWorld.take_action(current_state, action=action) for action in gridWorld.actions]

                # update the estimated value in gridWorld[x,y]
                v_update[x][y] = np.max(
                    np.multiply(act_prob, [reward + discnt_V * v_current[update_state[0], update_state[1]] \
                                             for [update_state, reward] in act_Results]))


                delta = max(delta, np.abs(v_current[x][y] - v_update[x][y]))

        # print(v_update)
        # print(opt_policy)
        # Plot expected value of all cell

    # Output a deterministic policy
    for x in range(9):
        for y in range(9):
            # when current state is 'X' or 'G', skip this iteration
            if GridWorld().map[x][y] == 'X' or GridWorld().map[x][y] == 'G':
                continue

            current_state = np.array([x, y])
            act_Results = [gridWorld.take_action(current_state, action=action) for action in gridWorld.actions]
            opt_policy[x][y] = np.argmax(np.multiply(act_prob, [reward + discnt_V * v_update[update_state[0], update_state[1]] \
                                                                  for [update_state, reward] in act_Results]))


    plot_policy(value_function=v_update, title="Exercise 3.3 Value Iteration Algorithm", num_action=num_action)
    
    plot_value(v_update, "Exercise 3.3 optimal value of all cells")

    print(* "Exercise 3.2 and 3.3 Comparison")
    print("Compare to policy iteration algorithm of 3.2, we can clearly see that value iteration algorithm of 3.3 converge much earlier than the algorithmn of 3.2 in terms of calculated optimal values.")
    print("Moreover, as it is allowed to move in diagonal direction in 3.3, the value iteration algorithm choose a diagonal action when it can reach to the higher value states or the goal quicker, instead of traveling around(indirect way) like the optimal policy of 3.2 does.")
    return opt_policy, v_update



# Exercise 3.4
def Non_deter_Value_Iteration(discnt_V, tolerance):
    gridWorld = GridWorld8D()

    # initialize variables for storing estimated value map
    # v_current = V(k), v_update = V(K+1)
    v_current = np.zeros([9, 9], dtype=float)
    v_update = np.zeros([9, 9], dtype=float)

    #up, down, right, left, upright, upleft, downright, downleft
    num_action = 8

    delta = sys.float_info.max

    # initialize act_prob
    act_prob = [1/num_action]*num_action

    while (delta > tolerance):
        # initialize difference between v_current and v_next as 0 in every sweep
        delta = 0.

        # initialize as -1 which denotes block:X, we do not take action in there, action index: [0:3]
        opt_policy=np.zeros([gridWorld.rowSize, gridWorld.colSize])-1

        # State = [x][y]
        # x index
        for x in range(9):
            # y index
            for y in range(9):
                v_current[x][y] = v_update[x][y]

                # when current state is 'X' or 'G', skip this iteration
                if GridWorld().map[x][y] == 'X' or GridWorld().map[x][y] == 'G':
                    continue

                current_state = np.array([x, y])
                act_Results = [gridWorld.take_action(current_state, action=action) for action in gridWorld.actions]

                # find desired direction
                desired_act_idx = np.argmax([reward + discnt_V * v_current[update_state[0], update_state[1]] \
                       for [update_state, reward] in act_Results])

                #print(desired_act_idx)

                #using clockwise order action list, generate the probability of being chosen each actions
                # generate probabilities of non-deterministic actions
                renewed_act_prob = []
                for i in range (num_action):
                    if i == desired_act_idx:
                        renewed_act_prob.append(0.7)

                    # 45 degree left or right
                    elif i == desired_act_idx-1 or i == (desired_act_idx+1)%num_action:
                        renewed_act_prob.append(0.15)
                    else:
                        renewed_act_prob.append(0)

                #print(renewed_act_prob)

                # update the estimated value in gridWorld[x,y]
                v_update[x][y] = np.max(
                    np.multiply([renewed_act_prob], [reward + discnt_V * v_current[update_state[0], update_state[1]] \
                                             for [update_state, reward] in act_Results]))

                opt_policy[x][y] = np.argmax(np.multiply([renewed_act_prob], [reward + discnt_V * v_current[update_state[0], update_state[1]] \
                                                                      for [update_state, reward] in act_Results]))

                delta = max(delta, np.abs(v_current[x][y] - v_update[x][y]))

        # print(v_update)
        # print(opt_policy)
        # Plot expected value of all cell
    plot_policy(value_function=v_update, title="Exercise 3.4 Value Iteration of Non-Deterministic actions",num_action=num_action)
    
    plot_value(v_update, "Exercise 3.4 optimal value of all cells")
    print(* "Exercise 3.3 and 3.4 Comparison")
    print("Compare to value iteration algorithm of 3.3, we can clearly see that value iteration algorithm of 3.4 converge much later than the algorithm of 3.3 in terms of calculated optimal values.")
    print("Moreover, as it is restricted by probability 0.7 to move to any desired direction in 3.4, the value iteration algorihmconverges much later than 3.3 where the probability of the agent moving to its desired direction was more than in 3.4.")
    return opt_policy, v_update



def plot_value(value_function, title):
    """
     plot the calculated value for each cell in a grid diagram
    :param value_function: calculated value
    :param title:
    :return: none
    """

    fig, ax = plt.subplots()

    plt.gca().invert_yaxis()

    value_function=np.round(value_function, decimals=2)
    for i,j in product(range(GridWorld().rowSize),repeat=2):

        if i !=GridWorld().rowSize and j !=GridWorld().rowSize:

            if value_function[j][i] == 0:
                #print(j, i)
                if i ==  7 and j == 6:

                    plt.text(0.2+ i,0.5+ j, 'G')
                else:
                    plt.text(0.2+ i,0.5+ j, 'X')
            else:
                plt.text(0.2+ i,0.5+ j, value_function[j][i])
    ax.set_yticks([x for x in range(10)], minor=False)
    ax.set_xticks([x for x in range(10)], minor=True)
    ax.yaxis.grid(True, which='both')
    ax.grid(True,which='both')
    plt.title(title)
    plt.show()


def plot_policy(value_function, title, num_action=None, policy=None):
    """
    plot the calculated optimal policy as a diagram with arrow
    """

    # for GridWorld with 4 actions
    if num_action==None:
        plt.figure()
        directions = []
        for x, y in product(range(9), repeat=2):
            if GridWorld().map[x, y] in ['X', 'G']:
                # 'X' and 'G' are not states
                continue
            # get action values
            act_Results = [GridWorld().take_action(np.array([x, y]), action)
                           for action in GridWorld().actions]
            values = [reward + 0.9 * value_function[state[0], state[1]] for state, reward in act_Results]

            max_val = max(values)
            optimal_act = np.where(np.array(values) == max_val)
            directions.extend([[x, y, act[0], act[1]] for act in
                               np.array(GridWorld().actions)[optimal_act]])


        #print(directions)
        directions = np.array(directions)
        plt.quiver(directions[:, 1], directions[:, 0], directions[
                                                       :, 3], -directions[:, 2], pivot='tail', scale=30, color='red')
        plt.ylim(9, -1)
        plt.xlim(-1, 9)
        plt.title(title)
        plt.show()

    # for GridWorld with 8 actions(diagonal)
    else:
        plt.figure()
        directions = []
        for x, y in product(range(9), repeat=2):
            if GridWorld8D().map[x, y] in ['X', 'G']:
                # 'X' and 'G' are not states
                continue

            # get action values
            act_Results = [GridWorld8D().take_action(np.array([x, y]), action)
                           for action in GridWorld8D().actions]
            values = [reward + 0.9 * value_function[state[0], state[1]] for state, reward in act_Results]

            max_val = max(values)
            optimal_act = np.where(np.array(values) == max_val)
            directions.extend([[x, y, act[0], act[1]] for act in
                               np.array(GridWorld8D().actions)[optimal_act]])


        directions = np.array(directions)
        plt.quiver(directions[:, 1], directions[:, 0], directions[
                                                       :, 3], -directions[:, 2], pivot='tail', scale=30, color='red')
        plt.ylim(9, -1)
        plt.xlim(-1, 9)
        plt.title(title)
        plt.show()

def main():
    # probability=[up, down, right, left], biased random action policy
    act_prob = [0.125, 0.625, 0.125, 0.125]

    # initialize discount value as 0.9
    discnt_V = 0.9
    # a enough small positive number which plays a role of determining when to stop the Iterative Policy Evaluation
    tolerance = 0.1

    # Exercise 3.1
    v = iter_policy_Evaluation(act_prob, discnt_V, tolerance)

    
    plot_value(v, "Exercise 3.1 expected value of all cells")


    # Exercise 3.2
    policy, v = policy_Iteration(act_prob, discnt_V, tolerance)
    # print(policy)

    # Exercise 3.3
    act_prob = [1/8]*8
    policy, v = Value_Iteration(act_prob, discnt_V, tolerance)
    #print(v)

    # Exercise 3.4
    policy, v = Non_deter_Value_Iteration(discnt_V,tolerance)

if __name__ == "__main__":
    main()
