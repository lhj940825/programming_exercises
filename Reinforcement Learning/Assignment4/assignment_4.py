"""
Robot Learning Exercise 4
Member 1
Name: Hojun Lim
Mat. No: 3279159

Member 2
Name: Kajaree Das
Mat. No: 3210311

"""
import matplotlib.pyplot as plt
from itertools import product
from random import choices
from tqdm import tqdm
import numpy as np
import sys


np.set_printoptions(suppress=True, precision=2)
#deviation from desired move
rightDeviation = np.array([[1, -1], [1, 1]])
leftDeviation = rightDeviation.T    

# GridWorld where state can move to 8 neighboring cell(direction)
class GridWorld(object):
    # define a 8 different action[up, down, right, left, upright, upleft, downright, downleft] that agent can take
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
        # define the grid world as [7x5] matrix as the question requires
        # G: goal, reward +1000p and the episode ends
        # W: reward -100p, episode ends
        # ' ': reward -1p, episode continues
        # diagonal move outside grid: vertical or horizontal move in the grid
        
        self.actions = [self.Action.up, self.Action.down, self.Action.right, self.Action.left, self.Action.upR, self.Action.upL, self.Action.doR, self.Action.doL]
        self.isStateFinal = False
        self.isStateStart = False
        self.map = np.array([[' ', ' ', ' ', 'W', 'W', 'G', ' '],
                             [' ', ' ', ' ', ' ', 'W', ' ', 'W'],
                             [' ', 'W', 'W', ' ', 'W', ' ', 'W'],
                             ['S', ' ', 'W', ' ', ' ', 'W', ' '],
                             [' ', ' ', 'W', ' ', ' ', ' ', ' ']])
        self.rowSize = self.map.shape[0]
        self.colSize = self.map.shape[1]
        self.mapSize = self.map.size
        
        
    def getNextAction(self):
        # chooses the next action probabilistically
        # returns next action or move
        actionIndices = [1, 2, 3] #indices representing desired action, left deviation, right deviation respectively
        actProbs = [0.6, 0.2, 0.2] #probabilities for choosing desired action, left deviation or right deviation respectively

        desiredActions = [2, 1, 0] #indices representing move to right, down and up
        desiredActProbs = [0.5, 0.25, 0.25] #probabilities for choosing to move to right, down and up
        action = np.random.choice(actionIndices, p=actProbs)
        desiredAct = np.random.choice(desiredActions, p=desiredActProbs)

        if action == 2:
            nextAction = np.round(np.dot(leftDeviation, self.actions[desiredAct]))
        elif action == 3:
            nextAction = np.round(np.dot(rightDeviation, self.actions[desiredAct]))
        else:
            nextAction = np.array(self.actions[desiredAct])
        return nextAction
        
        
    def getNextState(self, currentState, action):
        # input: current state(array containing coordinate) and action,
        # return: coordinate of the next state
        next_state = currentState + action
        next_state[1] = np.clip(next_state[1], 0, 6)
        next_state[0] = np.clip(next_state[0], 0, 4)
        return next_state
        
    
    def generateReward(self, currentState, next_state):
        # input: current state and next state,
        # return: acquired reward
        if ' ' in self.map[tuple(next_state)] or 'S' in self.map[tuple(next_state)]:
            return -1.
        elif 'W' in self.map[tuple(next_state)]:
            self.isStateFinal = True
            return -100.
        elif 'G' in self.map[tuple(next_state)]:
            self.isStateFinal = True
            return 1000.
        else:
            print('case return 0')
            return 0.
            
            
    def TDPolicyProbabilistic(self, alpha, gamma, numEpisodes):
        #alpha: step size or learning rate
        #gamma: discount rate
        
        # initialize variables for storing estimated value map
        # v_current = V(t), v_update = V(t+1)
        v_current = np.zeros([self.rowSize, self.colSize], dtype=float)
        for i in tqdm(range(numEpisodes)):
            self.isStateFinal = False
            state = np.array([3, 0]) # co-ordinate of the start state 'S'
            while True:
                action = self.getNextAction()
                next_state = self.getNextState(state, action)
                if self.isStateFinal: #terminal state reached
                    break
                reward = self.generateReward(state, next_state)
                v_current[state[0], state[1]] += alpha *(reward + gamma*v_current[next_state[0], next_state[1]] - v_current[state[0], state[1]])
                state = next_state
        return v_current
                
        
# Epsilon Greedy learning agent        
class EpsilonGreedyLearner(object):

    def __init__(self, epsilon, grid):
        self.epsilon = epsilon
        self.grid = grid
        self.Q = np.zeros((grid.rowSize, grid.colSize, len(grid.actions)))

    def getNextActionIndex(self, state):
        # chooses the next action using epsilon greedy policy
        # returns index of next action or move
        if np.random.rand() < self.epsilon:
            index = np.random.randint(8)
        else:
            index = np.argmax(self.Q[state[0], state[1], :])
        return index
       
    def getNextAction(self, desiredAct):
        # returns next action or move
        actProbs = [0.6, 0.2, 0.2]
        acts = [1, 2, 3]
        x = np.random.choice(acts, p=actProbs)
        if x == 2:
            nextAction = np.round(np.dot(leftDeviation, desiredAct))
        elif x == 3:
            nextAction = np.round(np.dot(rightDeviation, desiredAct))
        else:
            nextAction = desiredAct
        return nextAction
        
    def getNextState(self, currentState, action):
        # input: current state(array containing coordinate) and action,
        # return: coordinate of the next state
        next_state = currentState + action
        # check if we are stepping of the grid:
        next_state[1] = np.clip(next_state[1], 0, 6)
        next_state[0] = np.clip(next_state[0], 0, 4)
        return next_state
        
    # Chooses policy using Q-Learning
    def QLearningEpsilonGreedy(self, alpha, gamma, numEpisodes):
        #alpha: step size or learning rate
        #gamma: discount rate
        v_current = np.zeros([self.grid.rowSize, self.grid.colSize], dtype=float)
        for _ in tqdm(range(numEpisodes)):
            self.grid.isStateFinal = False
            state = np.array([3, 0])  # co-ordinate of the start state 'S'

            if _%1000 == 0:
                plot_policy(greedyLearner.Q, "Exercise 4.2 step:"+ str(_))
            while not self.grid.isStateFinal:
                actIndex = self.getNextActionIndex(state)
                action = self.getNextAction(self.grid.actions[actIndex])
                next_state = self.getNextState(state, action)
                reward = self.grid.generateReward(state, next_state)
                maxQ = np.max(self.Q[next_state[0], next_state[1], :])
                if self.grid.isStateFinal:
                    maxQ = 0
                v_current[state[0], state[1]] += alpha *(reward + gamma*v_current[next_state[0], next_state[1]] - v_current[state[0], state[1]])
                self.Q[state[0], state[1], actIndex] = ((1. - alpha) * self.Q[state[0], state[1], actIndex] + alpha * (reward + gamma * maxQ))
                state = next_state
        return v_current
        
def plot_policy(action_value, title):
    plt.figure()
    directions = []

    for x in range(5):
        for y in range(7):
            if GridWorld().map[x, y] in ['W', 'G']:
                # 'W' and 'G' are terminal states
                continue

            max_val = max(action_value[x][y])
            optimal_act = np.where(np.array(action_value[x][y]) == max_val)
            directions.extend([[x, y, act[0], act[1]] for act in
                               np.array(GridWorld().actions)[optimal_act]])


    directions = np.array(directions)
    plt.quiver(directions[:, 1], directions[:, 0], directions[
                                                   :, 3], -directions[:, 2], pivot='tail', scale=30, color='red')
    plt.ylim(5, -1)
    plt.xlim(-1, 7)
    plt.text(5,0, 'G')
    plt.title(title)
    plt.show()


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
    for i in range(GridWorld().colSize):
        for j in range(GridWorld().rowSize):
            if i !=GridWorld().colSize and j !=GridWorld().rowSize:

                if GridWorld().map[j, i] not in [' ']:
                    print(j,i)
                    if i ==  5 and j == 0:
                        plt.text(0.2+ i,0.5+ j, 'G')
                    elif i ==  0 and j == 3:
                        plt.text(0.2+ i,0.5+ j, 'S')
                    else:
                        plt.text(0.2+ i,0.5+ j, 'W')
                else:

                    plt.text(0.2+ i,0.5+ j, value_function[j][i])

    ax.set_yticks([x for x in range(GridWorld().rowSize+1)], minor=False)
    ax.set_xticks([x for x in range(GridWorld().colSize+1)], minor=True)
    ax.yaxis.grid(True, which='both')
    ax.grid(True,which='both')
    plt.title(title)
    fig.canvas.set_window_title(title)
    plt.show()




grid = GridWorld()

#Exercise 4.1
alpha = 0.01
gamma = 0.9
numEpisodes = 1000
vTD = grid.TDPolicyProbabilistic(alpha, gamma, numEpisodes)
print(vTD)
plot_value(vTD, 'TD(0)')

#Exercise 4.2
epsilon = 0.1
numEpisodes = 10000
alpha = 0.05
gamma = 0.85
greedyLearner = EpsilonGreedyLearner(epsilon, grid)
#assign Q values as per the probabilities of desired move
greedyLearner.Q[:, :, 2] = 0.5  #move right
greedyLearner.Q[:, :, 1] = 0.25 #move down
greedyLearner.Q[:, :, 0] = 0.25 #move up
vEGQ = greedyLearner.QLearningEpsilonGreedy(alpha, gamma, numEpisodes)
best_action = np.argmax(greedyLearner.Q, axis=2)
print(vEGQ)
plot_value(vEGQ, 'Epsilon Greedy Q-Learner')
print("best action:")
print(best_action)
plot_policy(greedyLearner.Q, "Exercise 4.2 step:10,000")
