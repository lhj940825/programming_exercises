#!/usr/bin/env python
# coding: utf-8

# In[40]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Robot Learning Exercise 5
Member 1
Name: Hojun Lim
Mat. No: 3279159

Member 2
Name: Kajaree Das
Mat. No: 3210311

"""
import numpy as np
import sys

#TODO add winner
WINNER = ['player', 'dealer', 'draw']
winnerIdx = {'player':0, 'dealer':1, 'draw':2}
class Player(object):
    def __init__(self):
        self.action = True
        self.score = 0
        self.isBusted = False
        self.point = 0
    #TODO: choose the action randomly    
    def makeChoice(self):
        actions = [True, False] #True: hit   False:Stick
        index = np.random.randint(2)
        self.action = actions[index]
        
    def checkIfBusted(self):
        if self.score > 21 or self.score < 1:
            self.isBusted = True
            
class Dealer(Player):
    def __init__(self):
        Player.__init__(self)
        self.firstCard = 0
        self.action = False
        
    def makeChoice(self):
        if self.score < 15:
            self.action = True
        else:
            self.action = False
        
        
class BeCareful(object):
    def __init__(self, player, dealer):
        self.player = player
        self.dealer = dealer
        self.isFinished = False
        self.isDraw = False
        self.states = []
        self.actions = []
        self.winner = None
        
    def setState(self):
        for i in range(21):
            self.states.append([self.dealer.firstCard, i+1, 0]) #defining state 's' (=i+1) as dealer's first card, player's score, n(s)
        if self.player.score <= 21:
            self.states[self.player.score - 1][2] += 1
    
    def setAction(self):
        for i in range(21):
            self.actions.append([0, 0]) #defining action 'a' as no of Hits, no of Sticks chosen for state i+1
    
        if self.player.score <= 21:
            if self.player.action:
                self.actions[self.player.score - 1][0] += 1
            else:
                self.actions[self.player.score - 1][1] += 1
    
    def draw(self):
        card = np.random.randint(low = 3, high=12)
        color = np.random.randint(low = 0, high = 10)
        return card, color
        
    def calculateScore(self, player, card, color):
        if 0 <= color <= 3:
            player.score -= card
        elif 4 <= color <= 10:
            player.score += card
        else:
            print("Wrong color")

    # TODO add winner
    def checkWinner(self):
        if self.dealer.isBusted:
            self.getReward(self.player, self.dealer, False)
            self.winner = WINNER[winnerIdx['player']]
        elif self.player.isBusted:
            self.getReward(self.dealer, self.player,  False)
            self.winner = WINNER[winnerIdx['dealer']]
        else:
            if self.player.score > self.dealer.score:
                self.getReward(self.player, self.dealer, False)
                self.winner = WINNER[winnerIdx['player']]
            elif self.dealer.score > self.player.score:
                self.getReward(self.dealer, self.player, False)
                self.winner = WINNER[winnerIdx['dealer']]
            else:
                self.getReward(self.dealer, self.player, True)
                self.winner = WINNER[winnerIdx['draw']]
            
    
    def checkIfFinished(self):
        if self.player.isBusted or self.dealer.isBusted:
            self.isFinished = True
            
    def getReward(self, winner, loser, isDraw):
        if not isDraw:
            winner.point += 1
            loser.point -= 1
        else:
            self.isDraw = True

    # TODO updated to return reward
    def advance(self,state, action):
        """

        :param state:
        :param action:
        :return:
        """
        #firstCard: dealer's first card
        #score: players's sum so far
        #action: player.hitMe; true for hit and false for a stick
        #print('Player Hit' if self.player.action==True else 'Player Stick')
        self.checkIfFinished()
        if self.isFinished:
            self.checkWinner()
        else:
            # when players action is hit
            if action:
                card, color = self.draw()
                self.calculateScore(self.player, card, color)
                self.player.checkIfBusted()
                # when player goes bust after the hit
                if self.player.isBusted:
                    self.checkWinner()


            # when player chose the 'stick'
            elif not action:
                self.dealer.checkIfBusted()

                # not actually choosing hit, just making decision
                self.dealer.makeChoice()
                #print('Dealer Hit' if self.dealer.action==True else 'Dealer Stick')
                # the case dealer chooses hit
                if not self.dealer.isBusted and self.dealer.action:
                    card, color = self.draw()
                    self.calculateScore(self.dealer, card, color)
                    self.dealer.checkIfBusted()

                # else: dealer chooses stick, so do nothing

                # going into the terminal phase, find who is the winner
                self.isFinished = True
                self.checkWinner()

            else:
                print("wrong action")


        if self.winner == 'player':
            reward = 1
        elif self.winner == 'dealer':
            reward = -1
        # when state is non-terminal or draw
        else:
            reward = 0
        playersNextState = [self.dealer.firstCard, self.player.score]
        return playersNextState, reward
    
    def firstDraw(self):
        playerCard = np.random.randint(low = 3, high=12)
        dealerCard = np.random.randint(low = 3, high=12)
        self.dealer.firstCard = dealerCard
        self.player.score += playerCard
        self.dealer.score += dealerCard
        self.player.makeChoice()
            
    def play(self):
        self.firstDraw()
        playersNextState = self.player.score
        print('player:' + str(self.player.score) +  " dealer:" + str(self.dealer.score)  )
        self.winner = None

        while not self.isFinished:
            playersNextState, reward = self.advance(self.player.score, self.player.action)
            print(reward)
            self.checkIfFinished()
            self.player.makeChoice()
            print('player:' + str(self.player.score) +  " dealer:" + str(self.dealer.score)  )

        print('Game Over')
        return playersNextState, self.player.point
            
        
            
            

def main():

    player = Player()
    dealer = Dealer()
    game = BeCareful(player, dealer)
    game.play()
    if game.isDraw:
        print("Its a draw!")
    elif game.player.point == 1:
        print("The winner is the player with " + str(game.player.score) + " points.")
    elif game.dealer.point == 1:
        print("The winner is the dealer with " + str(game.dealer.score) + " points.")
    else:
        print(game.dealer.point)
        print(game.player.point)
        print('goes wrong')



if __name__ == "__main__":
    main()

