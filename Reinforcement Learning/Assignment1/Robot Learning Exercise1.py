#!/usr/bin/env python
# coding: utf-8

#Solution for Assignment 1        

# Member 1
# Name: Hojun Lim
# Mat. No: 3279159
# 
# Member 2
# Name: Kajaree Das
# Mat. No: 3210311
 
#        

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

num_arms = 6
arms ={"First":0, "Second":1, "Third":2, "Fourth":3, "Fifth":4, "Sixth":5}


# In[2]:


# get the value of reward based on chosen action
def sample_Reward(action, iter = None):
    if iter == None:
        if action == arms["First"]:
            return(np.random.uniform(1,3))
        elif action == arms["Second"]:
            return(np.random.uniform(-3,8))
        elif action == arms["Third"]:
            return(np.random.uniform(2,5))
        elif action == arms["Fourth"]:
            return(np.random.uniform(-2,6))
        elif action == arms["Fifth"]:
            return(np.random.uniform(3,4))
        # case of sixth arm
        else:
            return(np.random.uniform(-2,2))
    
    # code for exercise 1.4)
    else:
        if action == arms["First"]:
            return(np.random.uniform(1,3))
        elif action == arms["Second"]:
            return(np.random.uniform(-3,8))
        elif action == arms["Third"]:
            return(np.random.uniform(2,5))
        elif action == arms["Fourth"]:
            # After 2000 steps, sample the rewards of the fourth arm uniformly from [5,7)
            if iter >=2000:
                return (np.random.uniform(5,7))
            else:
                return(np.random.uniform(-2,6))
        elif action == arms["Fifth"]:
            return(np.random.uniform(3,4))
        # case of sixth arm
        else:
            return(np.random.uniform(-2,2))
        
def plot_Average_Reward(rewards, title, comparison=None):
    
    # plot one line
    if comparison == None:
        # calculate the average rewards per steps
        rewards = np.cumsum(rewards)/ np.arange(1, len(rewards)+1, dtype=np.float32)

        plt.plot(rewards)
        plt.xlabel("step")
        plt.ylabel("average reward")
        plt.title(title)
        plt.show()
    
    # plot two line for comparison (Exercise 1.4 and 1.5)
    else :
        Ex4_rewards = np.cumsum(rewards[0])/ np.arange(1, len(rewards[0])+1, dtype=np.float32)
        Ex5_rewards = np.cumsum(rewards[1])/ np.arange(1, len(rewards[0])+1, dtype=np.float32)
        
        plt.plot(Ex4_rewards)
        plt.plot(Ex5_rewards)
        plt.xlabel("step")
        plt.ylabel("average reward")
        plt.title(title)
        plt.legend(['Epsilon=0.1, a=0.01 (Ex 1.4)', 'Greedy, a=0.01 (Ex 1.5)'])
        plt.show()
        
         
    
def plot_Optimal_Action(actions, title):
    
    # calculate the percentage of each actions per steps
    for i in range(len(actions[:,])):
        action = np.cumsum(actions[i,:])/np.arange(1, len(rewards)+1, dtype=np.float32)
        actions[i,:] = action
        plt.plot(actions[i,:])
        
    plt.xlabel("step")
    plt.ylabel("percentage of actions")
    plt.title(title)
    plt.legend(['Arm 1','Arm 2','Arm 3','Arm 4','Arm 5','Arm 6'])
    plt.show()
    


# In[3]:


#  Exercise 1.2)

num_play = 10
# assume the default value of Q(Expected Reward) is 0
Q = [0.]*num_arms

# variable for storing the history of rewards from chosen arms(actions)
rewards_list = []
for _ in range(num_arms):
    rewards_list.append([])
    
for iter in range(0,num_play):
    # choose actions uniformly(in range of [0,6) = total 6 actions)
    chosen_action = np.random.randint(0,6) 
    # store the generated rewards 
    rewards_list[chosen_action].append(sample_Reward(chosen_action)) 
    

# convert 2d list to numpy array
rewards_list = np.array([np.array(rewards) for rewards in rewards_list])

# update the expected value of rewards of a(action)
for a in range(0, num_arms):
    # when the a-th arm(action) has been chosen more than once
    if np.size(rewards_list[a]) != 0:
        # calculate Q by 'Sample-Average Method'
        Q[a] = np.mean(rewards_list[a])
        
# Visualize the outcome 
plt.plot(Q)
plt.xlabel("X-th Action(Arm)")
plt.ylabel("Q(a)")
plt.title("Exercise 1.2")
plt.show()

# Print
print("Solution for 1.2")
print("-----------------")
for a in range(0, num_arms):
    print('Calculated sample average reward of {}-th action is : {}'.format(str(a), str(Q[a])))
    
print("Sample Average Reward in Total : {}".format(np.mean(Q)))
    

# In[4]:


# Exercise 1.3)


num_play = 4000

# initialize Q(ai)=0
Q = [0.]*num_arms
# variable for storing counts when the respective action is chosen
action_CNT = [0]*num_arms
epsilon = 0.1

# variable for computing average rewards
rewards = []

# variable for traking history of chosen action per steps
actions = np.zeros([num_arms, num_play], dtype = np.float64)

print("Solution for 1.3")
print("-----------------")

for iter in range(1,num_play+1):
    # choose actions uniformly with the chance of epsilon
    if np.random.uniform(0,1) > (1-epsilon):
        chosen_action = np.random.randint(0,6)
    # choose optimum action
    else:
        chosen_action = np.argmax(Q)
    
    # increase the count
    action_CNT[chosen_action] += 1
    

    # save the chosen action history 
    actions[chosen_action][iter-1] = 1

    reward = sample_Reward(chosen_action)
    rewards.append(reward)
    
    # update the action value(expected reward) by computing recursively
    Q[chosen_action] = Q[chosen_action]+(reward-Q[chosen_action])/(action_CNT[chosen_action])
    
    if iter % 100 == 0:  
        print("In iter {} \n*Percentage of choosing ".format(str(iter)),end ='')
        
        for a in range(0,6):
            percent = action_CNT[a]/iter*100
            print(" Arm {}: {}%".format(a+1, percent), end ='')    
            
        print()
        print("*Average Reward : {}".format(np.mean(np.array(rewards)), end = ''))
        
plot_Average_Reward(rewards, "Exercise 1.3")
plot_Optimal_Action(actions, "Exercise 1.3")


# In[5]:


# Exercise 1.4)


num_play = 4000

# initialize Q(ai)=0
Q = [0.]*num_arms
# variable for storing counts when the respective action is chosen
action_CNT = [0]*num_arms
epsilon = 0.1

# constant learning rate a
const_a = 0.01
rewards = []

# variable for traking history of chosen action per steps
actions = np.zeros([num_arms, num_play], dtype = np.float64)

print("Solution for 1.4")
print("-----------------")

for iter in range(1,num_play+1):
    # choose actions uniformly with the chance of epsilon
    if np.random.uniform(0,1) > (1-epsilon):
        chosen_action = np.random.randint(0,6)
    # choose optimum action
    else:
        chosen_action = np.argmax(Q)
    
    # increase the count
    action_CNT[chosen_action] += 1

    # save the chosen action history 
    actions[chosen_action][iter-1] = 1

    reward = sample_Reward(chosen_action)
    rewards.append(reward)
    
    # Non-stationary
    # update the action value(expected reward) by computing recursively with learning rate a(=0.01)
    Q[chosen_action] = Q[chosen_action]+const_a*(reward - Q[chosen_action])
    
    
    if iter % 100 == 0:  
        print("In iter {} \n*Percentage of choosing ".format(str(iter)),end ='')
        
        for a in range(0,6):
            percent = action_CNT[a]/iter*100
            print(" Arm {}: {}%".format(a+1, percent), end ='')    
            
        print()
        print("*Average Reward : {}".format(np.mean(np.array(rewards)), end = ''))
        
plot_Average_Reward(rewards, "Exercise 1.4")
plot_Optimal_Action(actions, "Exercise 1.4")

# save rewards for comparison with rewards from ex1.5
Ex4_rewards= rewards

# In[6]:


# Exercise 1.5)


num_play = 4000

# initialize Q(ai)=5
Q = [5.]*num_arms
# variable for storing counts when the respective action is chosen
action_CNT = [0]*num_arms
epsilon = 0.1
# constant learning rate a
const_a = 0.01
rewards = []
# variable for traking history of chosen action per steps
actions = np.zeros([num_arms, num_play], dtype = np.float64)

print("Solution for 1.5")
print("-----------------")


for iter in range(1,num_play+1):

    # choose only optimum action(Greedy)
    chosen_action = np.argmax(Q)
    
    # increase the count
    action_CNT[chosen_action] += 1
    
    # save the chosen action history 
    actions[chosen_action][iter-1] = 1
    
    reward = sample_Reward(chosen_action)
    rewards.append(reward)
    
    # Non-stationary
    # update the action value(expected reward) by computing recursively with learning rate a(=0.01)
    Q[chosen_action] = Q[chosen_action]+const_a*(reward - Q[chosen_action])

    
    if iter % 100 == 0:  
        print("In iter {} \n*Percentage of choosing ".format(str(iter)),end ='')
        
        for a in range(0,6):
            percent = action_CNT[a]/iter*100
            print(" Arm {}: {}%".format(a+1, percent), end ='')    
            
        print()
        print("*Average Reward : {}".format(np.mean(np.array(rewards)), end = ''))
        
plot_Average_Reward(rewards, "Exercise 1.5")    
plot_Optimal_Action(actions, "Exercise 1.5")

# save rewards for comparison with rewards from ex1.4
Ex5_rewards = rewards


# In[7]:


plot_Average_Reward([Ex4_rewards, Ex5_rewards], "Comparision of Exercise 1.4, 1.5", True)


# In[ ]:
