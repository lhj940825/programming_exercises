#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Robot Learning Exercise 5
Member 1
Name: Hojun Lim
Mat. No: 3279159

Member 2
Name: Kajaree Das
Mat. No: 3210311

"""


import matplotlib.pyplot as plt

from beCareful4CoarseCoding import Player, Dealer, BeCareful
import itertools as it
import numpy as np

Action = {'hit':0, 'stick':1}

dealer_features = [(3, 6), (6, 9), (9, 12)]
player_features = [(1, 6), (4, 9), (7, 12), (10, 15), (13, 18), (16, 21)]
action_features = [Action['hit'], Action['stick']]
feature_space = list(it.product(dealer_features, player_features, action_features))

class LinearF_Approximator:
    def __init__(self, _lambda, traning_iter=1000, test_iter=100, exp_prob=0.05, alpha=0.01, gamma=0.9, becareful = None):
        self._lambda = _lambda
        self.traning_iter = traning_iter
        self.test_iter = test_iter
        self.exp_prob = exp_prob
        self.alpha = alpha
        self.gamma = gamma
        self.theta = np.array([0.]*len(feature_space))
        self.e_trace = np.array([0.]*len(feature_space))
        self.becareful = becareful
        #self.learning_curve = []

    def get_featureVector(self, state, action):
        feature_vector = list()
        dealer_card, player_sum = state

        def contains(i, v):
            l, u = i
            return l <= v <= u

        def feature_on(f):
            d, p, act = f
            return contains(d, dealer_card) and contains(p, player_sum) and act == action

        for (i, f) in enumerate(feature_space):
            if feature_on(f):
                feature_vector.append(1)
            else:
                feature_vector.append(0)
        return feature_vector

    def _phi(self,state , action):

        dealer_card, player_sum = state

        def contains(i, v):
            l, u = i
            return l <= v <= u

        def feature_on(f):
            d, p, act = f
            return contains(d, dealer_card) and contains(p, player_sum) and act == action


        return [i for (i, f) in enumerate(feature_space) if feature_on(f)]

    def q(self, state, act):
        f_a = self._phi(state, act)
        return sum([self.theta[i] for i in f_a])

    # E-Greedy
    def make_decision(self, state):

        # Exploring state
        if np.random.rand() < self.exp_prob:
            return Action['hit'] if np.random.rand() < 0.5 else Action['stick']


        # else: choose in a greedy way with regard to theta
        else:

            feature_vectors = [self.get_featureVector(state, Action['hit']),self.get_featureVector(state, Action['stick']) ]

            q_values = [np.matmul(vector, self.theta.transpose()) for vector in feature_vectors]
            action = np.argmax(q_values)

            print('action:' + str(action))
            return action


    # for testing
    def make_greedy_decision(self, state):
        feature_vectors = [self.get_featureVector(state, Action['hit']),self.get_featureVector(state, Action['stick']) ]

        q_values = [np.matmul(vector, self.theta.transpose()) for vector in feature_vectors]
        action = np.argmax(q_values)

        print('action:' + str(action))
        return action


    def update_theta(self, TD_error):
        for i in range(len(feature_space)):
            #print('val theta before:'+str(self.theta[i]))
            self.theta[i] = (self.theta[i] + self.alpha * TD_error * self.e_trace[i])

            #print('theta' + str(self.alpha * TD_error * self.e_trace[i]) + 'theta val '+ str(self.theta[i] + self.alpha * TD_error * self.e_trace[i]))

    def train(self):
        iter = 0

        # number of episodes
        while iter < self.traning_iter:
            # eligibility trace clear
            self.e_trace = np.array([0.]*len(feature_space))

            # state = (dealer card number, player card sum)
            state = self.becareful.generate_initial_state()
            # until the episode get into the terminal state
            while not self.becareful.isFinished:
                action = self.make_decision(state)
                next_state, reward = self.becareful.advance(state, action, 'coarse coding')

                # get corresponding feature vector for given state and action
                current_state_feature_vector = self.get_featureVector(state, action)
                next_state_feature_vector = self.get_featureVector(next_state, self.make_decision(next_state))

                self.e_trace = (self.gamma*self._lambda*self.e_trace + current_state_feature_vector)

                # Q(s,a) = feature_vector*theta(transposed)
                TD_error = reward + self.gamma*np.matmul(next_state_feature_vector, self.theta.transpose()) - np.matmul(current_state_feature_vector, self.theta.transpose())

                self.update_theta(TD_error)

                state = next_state



            print('one game end')
            iter += 1

        return self

    def test(self):
        accumulated_reward = []
        iter = 0
        # number of episodes
        while iter < self.test_iter:
            rewards = []
            # eligibility trace clear
            self.e_trace = np.array([0.]*len(feature_space))

            # state = (dealer card number, player card sum)
            state = self.becareful.generate_initial_state()

            # until the episode get into the terminal state
            while not self.becareful.isFinished:
                action = self.make_greedy_decision(state)
                next_state, reward = self.becareful.advance(state, action, 'coarse coding')
                rewards.append(reward)
                state = next_state


                print(reward)
            print('one game end')
            iter += 1
            print(rewards)


            accumulated_reward.append(sum(rewards))

        return accumulated_reward

def plot_graph(rewards_list, test_iter, lambdas):
    x = np.arange(0, test_iter)+1

    for i,l  in enumerate(lambdas):
        plt.plot(x, rewards_list[i])


    plt.legend(['lambda = '+str(round(l,2)) for l in lambdas], loc='lower left')
    plt.xlabel('step')
    plt.ylabel('accumulated reward')
    plt.title('5.3) Sarsa with linear value function approximation')
    plt.show()

def plot_final_reward_graph(rewards_list, test_iter, lambdas):
    fig, ax = plt.subplots()
    final_rewards = [rewards[test_iter-1] for rewards in rewards_list ]
    ax.plot(lambdas, final_rewards)

    ax.set(xlabel='lambda', ylabel='accumulated(final) reward',
           title='SARSA-Linear Value Function Approximation')
    ax.grid()

    plt.show()


def main():

    rewards_by_Lambda = []
    lambdas = np.arange(0, 1.1, 0.1)
    for l in lambdas:
        player = Player()
        dealer = Dealer()
        game = BeCareful(player, dealer)
        agent = LinearF_Approximator(_lambda=l, traning_iter=1000, test_iter=100, exp_prob=0.1, alpha=0.05, gamma=0.5, becareful=game)
        agent.train()
        accumulated_rewards = agent.test()
        accumulated_rewards = np.array(accumulated_rewards).cumsum()
        rewards_by_Lambda.append(accumulated_rewards)

    plot_graph(rewards_by_Lambda, test_iter=100, lambdas = lambdas)
    plot_final_reward_graph(rewards_by_Lambda, test_iter=100, lambdas=lambdas)

if __name__ == "__main__":
    main()


