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
        self.winner = None
    
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


    def checkWinner(self):
        if self.dealer.isBusted:
            self.getReward(self.player, self.dealer, False)
            self.winner = WINNER[winnerIdx['player']]
            self.isFinished = True
        elif self.player.isBusted:
            self.getReward(self.dealer, self.player,  False)
            self.winner = WINNER[winnerIdx['dealer']]
            self.isFinished = True
        else:
            if self.player.score > self.dealer.score:
                self.getReward(self.player, self.dealer, False)
                self.winner = WINNER[winnerIdx['player']]
                self.isFinished = True
            elif self.dealer.score > self.player.score:
                self.getReward(self.dealer, self.player, False)
                self.winner = WINNER[winnerIdx['dealer']]
                self.isFinished = True
            else:
                self.getReward(self.dealer, self.player, True)
                self.winner = WINNER[winnerIdx['draw']]
                self.isFinished = True
            
    
    def checkIfFinished(self):
        if self.player.isBusted or self.dealer.isBusted:
            self.isFinished = True
            
    def getReward(self, winner, loser, isDraw):
        if not isDraw:
            winner.point += 1
            loser.point -= 1
        else:
            self.isDraw = True

    def advance(self,state, action, type=None):
        """

        :param state: [dealer's first card, players's sum so far]
        :param action: player.hitMe; true for hit and false for a stick
        :return: nextState, reward
        """
        
        if type == None:
            print('Player Hit' if self.player.action==True else 'Player Stick')
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
                    print('Dealer Hit' if self.dealer.action==True else 'Dealer Stick')
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
            playersNextState = self.player.score
            return playersNextState, reward

        # when coarse coding class calls this method
        else:

            #print('Player Hit' if self.player.action==True else 'Player Stick')
            self.checkIfFinished()
            if self.isFinished:
                self.checkWinner()

                print('1234')
                return None, 0
            else:
                # when players action is hit
                if action:
                    print("player Hit")
                    card, color = self.draw()
                    self.calculateScore(self.player, card, color)
                    self.player.checkIfBusted()
                    print('after player hit'+ str(self.player.score) + " "+ str(self.dealer.score))

                    # when player goes bust after the hit
                    if self.player.isBusted:
                        self.checkWinner()
                        self.isFinished = True

                        print('after hit player burst'+ str(self.player.score)+ " "+ str(self.dealer.score))


                # when player chose the 'stick'
                elif not action:
                    self.dealer.checkIfBusted()
                    print('player stick'+ str(self.player.score)+ " "+ str(self.dealer.score))
                    # not actually choosing hit, just making decision
                    self.dealer.makeChoice()
                    print('Dealer Hit' if self.dealer.action==True else 'Dealer Stick')
                    # the case dealer chooses hit
                    if not self.dealer.isBusted and self.dealer.action:
                        card, color = self.draw()
                        self.calculateScore(self.dealer, card, color)
                        self.dealer.checkIfBusted()

                    # else: dealer chooses stick, so do nothing
                    print('after dealer dicision'+ str(self.player.score)+ " "+ str(self.dealer.score))
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
            playersNextState = (self.dealer.score, self.player.score)
            return playersNextState, reward

    
    def firstDraw(self):
        playerCard = np.random.randint(low = 3, high=12)
        dealerCard = np.random.randint(low = 3, high=12)
        self.dealer.firstCard = dealerCard
        self.player.score += playerCard
        self.dealer.score += dealerCard
        self.player.makeChoice()

    
    def generate_initial_state(self):
        """
        initialize all the setting for playing new game
        :return:
        """
        self.isFinished = False
        playerCard = np.random.randint(low = 3, high=12)
        dealerCard = np.random.randint(low = 3, high=12)

        self.player.score = 0
        self.dealer.score = 0
        self.player.isBusted = False
        self.dealer.isBusted = False

        self.isFinished = False
        self.isDraw = False
        self.states = []
        self.actions = []
        self.winner = None

        self.dealer.firstCard = dealerCard
        self.player.score += playerCard
        self.dealer.score += dealerCard
        self.player.makeChoice()

        return dealerCard, playerCard

            
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
