"""
Robot Learning Exercise 5
Member 1
Name: Hojun Lim
Mat. No: 3279159

Member 2
Name: Kajaree Das
Mat. No: 3210311

"""
from beCareful import Player, Dealer, BeCareful
import matplotlib.pyplot as plt
import numpy as np
import sys


hit = 0
stick = 1    
    
class Sarsa(BeCareful):
    def __init__(self, lamda, player, dealer):
        BeCareful.__init__(self, player, dealer)
        self.lamda = lamda
        self.epsilon = 1.0
        self.actions = [0, 1] #True: hit False: stick
        self.e = np.zeros([10, 21, 2])
        self.Q = np.random.randn(10, 21, 2)
        self.isTerminal = False
        self.N = np.zeros([10, 21, 2])
        self.numExplore = 0
        self.greedyReward = 0
    
    def setEpsilon(self, state, action):
        nOffset = 10
        if self.numExplore > 1000:
            self.epsilon = 1
        else:
            self.epsilon = nOffset/(nOffset+self.N[state[0]-3][self.player.score -1][action])
    
    def chooseAction(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(2) #1: hit 0: stick
        if self.Q[state[0]-3][state[1]-1][0] == self.Q[state[0]-3][state[1]-1][1]:
            return np.random.randint(2)
            #index = np.argmax(self.Q[dealerFirstCard-3, state[1], :]) #replace with greedy strategy
        if self.Q[state[0]-3][state[1]-1][0] > self.Q[state[0]-3][state[1]-1][1]:
            return 0
        else:
            return 1
    
    def checkIfTerminal(self, nextStateIdx):
        if nextStateIdx > 21:
            self.isTerminal = True
        self.checkIfFinished()
        if self.isFinished:
            self.isTerminal = True
            
    def trainAgent(self, numEpisodes):
        while self.numExplore < numEpisodes:
            self.e = np.zeros((10, 21, 2))
            self.player = Player()
            self.dealer = Dealer()
            self.firstDraw()
            state = [self.dealer.firstCard, self.player.score]
            action = self.chooseAction(state)
            self.runEpisode(state, action)
            self.isTerminal = False
            self.numExplore += 1
        self.isTerminal = False
        
            
    def runTest(self):
        accumulated_reward = []
        actions = [True, False] #True: hit   False:Stick
        for i in range(100):
            rewards = []
            self.player = Player()
            self.dealer = Dealer()
            self.firstDraw()                
            state = [self.dealer.firstCard, self.player.score]
            actionIdx = self.chooseAction(state)
            self.isTerminal = False
            
            while not self.isTerminal:
                nextState, reward = self.advance(state, actions[actionIdx])
                rewards.append(reward)
                self.checkIfTerminal(state[1])
                if self.isTerminal:
                    break
                self.setEpsilon(nextState, actionIdx)
                nextActionIdx = self.chooseAction(nextState)
                state = nextState
                actionIdx = nextActionIdx
            accumulated_reward.append(sum(rewards))
        return accumulated_reward

        
    def runEpisode(self, state, actionIdx):
        gamma = 1
        actions = [True, False] #True: hit   False:Stick
        while not self.isTerminal:
            nextState, reward = self.advance(state, actions[actionIdx])
            if self.numExplore > 1000:
                print("End of learning")
                break
            self.checkIfTerminal(state[1])
            if self.isTerminal:
                break
            self.setEpsilon(nextState, actionIdx)
            nextActionIdx = self.chooseAction(nextState)
            delta = reward + gamma*self.Q[nextState[0]-3][nextState[1]-1][nextActionIdx] - self.Q[state[0]-3][state[1]-1][actionIdx]
            self.e[state[0]-3][state[1]-1][actionIdx - 1] += 1
            self.N[state[0]-3][state[1]-1][actionIdx] += 1
            alpha = 1./self.N[state[0]-3][state[1]-1][actionIdx]
            for s in range(21):
                for a in range(2):
                    self.Q[nextState[0]-3][s][a] += alpha*delta*self.e[nextState[0]-3][s][a]
                    self.e[nextState[0]-3][s][a] *= gamma*self.lamda
            state = nextState
            actionIdx = nextActionIdx
            
            
def plot_graph(rewards_list, test_iter, lambdas):
    x = np.arange(0, test_iter)+1

    for i,l  in enumerate(lambdas):
        plt.plot(x, rewards_list[i])


    plt.legend(['lambda = '+str(round(l,2)) for l in lambdas], loc='lower left')
    plt.xlabel('step')
    plt.ylabel('accumulated reward')
    plt.title('5.2 SARSA-Lambda: Accumulated Reward for different Lambdas')
    plt.show()

def plot_final_reward_graph(rewards_list, test_iter, lambdas):
    fig, ax = plt.subplots()
    final_rewards = [rewards[test_iter-1] for rewards in rewards_list ]
    ax.plot(lambdas, final_rewards)

    ax.set(xlabel='lambda', ylabel='accumulated(final) reward',
           title='5.2 SARSA-Lambda: Accumulated Reward vs Lambda')
    ax.grid()

    plt.show()

lambdas = np.arange(0, 1.1, 0.1)
rewards = []
numEpisodes = 1000
for l in lambdas:
    player = Player()
    dealer = Dealer()
    agent = Sarsa(l, player, dealer)
    agent.trainAgent(numEpisodes)
    accumulated_rewards = agent.runTest()
    accumulated_rewards = np.array(accumulated_rewards).cumsum()
    rewards.append(accumulated_rewards)
    

plot_graph(rewards, test_iter=100, lambdas = lambdas)
plot_final_reward_graph(rewards, test_iter=100, lambdas=lambdas)
