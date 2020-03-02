import numpy as np
import matplotlib.pyplot as plt

# import sys
# sys.setrecursionlimit(10000)




# list of information about tasks, form: (task number(state), rewards, probability of passing)
taskList = [['1', 12, 0.25], ['1F', 12.0, 0.125], ['2', 4, 0.4], ['2F', 4, 0.2], ['3', 10, 0.35], ['3F', 10, 0.175], ['4', 5, 0.6], ['4F', 5, 0.3], ['5', 7, 0.45], ['5F', 7, 0.225],  ['6', 3, 0.5], ['6F', 3, 0.25], ['7', 50, 0.15], ['7F', 50, 0.075]]


attribute = {"probabilty":0, "score":1}

#
case_list = []
def cal_prob(probability, score, taskList, attempt):

    # when students use all chances(attempts) or finish taking exam
    if len(taskList) <= 0 or attempt >=10:
        # append at list
        case_list.append([probability, score, attempt])
        return 0

    # when students are taking exam
    else:

        # retrieve current task(question) from task list
        currentTask = taskList[0]

        # case: if current task is a second attempt
        if len(currentTask[0]) == 2: # when the shape of task ID(currentTask[0]) is 'XF'

            # case of solving task
            cal_prob(probability*currentTask[2], score+currentTask[1], taskList[1:], attempt+1)

            # case of failure at solving task
            cal_prob(probability*(1-currentTask[2]), score, taskList[1:], attempt+1)

        else:
            # if currentTask is on first attempt and is solved then move to two-step forward state in sequantialPolicy (which means from 'X' to 'X+1')\
            # otherwise move one step forward (from state 'X' to 'XF')

            # case of solving task
            cal_prob(probability*currentTask[2], score+currentTask[1], taskList[2:], attempt+1)

            # case of failure at solving task
            cal_prob(probability*(1-currentTask[2]), score, taskList[1:], attempt+1)


# variables for calculating
init_prob = 1
init_score = 0
init_attempt = 0
cal_prob(init_prob, init_score, taskList, init_attempt)


print("all possible cases[probability, score, number of attempts]:")
print(case_list)
sum_prob = 0
for i in case_list:
    sum_prob+= i[0]
print("Sum of probabilities in each cases: {}".format(sum_prob))
print("total number of possible cases: {}".format(len(case_list)))

# variable to store the probability of passing the exam
prob_pass = 0
max_score = 0 # maximum score that students can get

for task in taskList:

    # retrieve score information from task
    if len(task[0]) == 1:
        max_score+=task[1]

print("Maximum score is: {}".format(max_score))

for case in case_list:
    if case[attribute["score"]] >= (max_score/2.0):
        prob_pass+= case[attribute["probabilty"]]

print("The probability of passing the exam is: {}%".format(prob_pass*100))