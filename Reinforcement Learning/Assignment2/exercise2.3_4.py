import numpy as np
import matplotlib.pyplot as plt

# import sys
# sys.setrecursionlimit(10000)




# list of information about tasks, form: (task number(state), rewards, probability of passing)
#taskList = [['1', 12, 0.25], ['1F', 12.0, 0.125], ['2', 4, 0.4], ['2F', 4, 0.2], ['3', 10, 0.35], ['3F', 10, 0.175], ['4', 5, 0.6], ['4F', 5, 0.3], ['5', 7, 0.45], ['5F', 7, 0.225],  ['6', 3, 0.5], ['6F', 3, 0.25], ['7', 50, 0.15], ['7F', 50, 0.075]]

# define the tasks and set its propertys
passedTask = [['1', 12, 0.25],  ['2', 4, 0.4],  ['3', 10, 0.35], ['4', 5, 0.6], ['5', 7, 0.45],  ['6', 3, 0.5], ['7', 50, 0.15]]
failedTask = [['7F', 50, 0.075], ['6F', 3, 0.25], ['5F', 7, 0.225], ['4F', 5, 0.3], ['3F', 10, 0.175],['2F', 4, 0.2],['1F', 12.0, 0.125]]

# value to store all performance per policies
performance = []

# type of policies
policies = ["sequential", "increasing difficulty", "increasing reward", "decreasing difficulty", "decreasing reward"]
idx_policy = {"sequential":0, "increasing difficulty":1, "increasing reward":2, "decreasing difficulty":3, "decreasing reward":4}
#list.sort(key=lambda taskProperty:taskProperty[0])



# Calculate Expected return Value using Bellman Equation
def expectedState_Value(taskList):
    if len(taskList) > 0:

        currentTask = taskList[0]


        # case: if current task is a second attempt
        if len(currentTask[0]) == 2: # when the shape of task ID(currentTask[0]) is 'XF'

            # if currentTask is on second attempt then whether it is solved or not, only move one state forward (from state 'XF' to 'X+1')
           value = currentTask[2]*(currentTask[1] + expectedState_Value(taskList[1:])) \
                    + (1-currentTask[2])*(0+ expectedState_Value(taskList[1:]))

        # case: if current task is a first attempt
        else:
            # if currentTask is on first attempt and is solved then move to two-step forward state in sequantialPolicy (which means from 'X' to 'X+1')\
            # otherwise move one step forward (from state 'X' to 'XF')
            value = currentTask[2]*(currentTask[1] + expectedState_Value(taskList[2:])) \
                     + (1-currentTask[2])*(0+ expectedState_Value(taskList[1:]))

        return value
    else:
        return 0


def choosePolicy(key):
    if key == "sequential":

        # x[0]: task id(number)
        taskList = sorted(passedTask, key= lambda x:x[0])

        for task in failedTask:

            idx = 0
            for j in range(len(taskList)):

                # find appropriate index to insert
                if taskList[j][0] == task[0][0]:
                    idx = j

            taskList.insert(idx+1, task)
        print(taskList)
        return taskList

    elif key == "increasing difficulty":

        # x[2]: probability of first attempt solution, sorting in the order of increasing difficulty
        taskList = sorted(passedTask, key= lambda x:x[2], reverse=True)
        for task in failedTask:
            idx = 0
            for j in range(len(taskList)):

                # find appropriate index to insert
                if taskList[j][0] == task[0][0]:
                    idx = j

            taskList.insert(idx+1, task)
        print(taskList)
        return taskList


    elif key == "increasing reward":

        # x[2]: rewards, sorting in the order of increasing reward
        taskList = sorted(passedTask, key= lambda x:x[1])
        for task in failedTask:
            idx = 0
            for j in range(len(taskList)):
                if taskList[j][0] == task[0][0]:
                    idx = j

            taskList.insert(idx+1, task)
        print(taskList)
        return taskList

    elif key == "decreasing difficulty":

        # x[2]: probability of first attempt solution, sorting in the order of decreasing difficulty
        taskList = sorted(passedTask, key= lambda x:x[2])
        for task in failedTask:
            idx = 0
            for j in range(len(taskList)):
                if taskList[j][0] == task[0][0]:
                    idx = j

            taskList.insert(idx+1, task)
        print(taskList)
        return taskList

    elif key == "decreasing reward":

        # x[2]: rewards, sorting in the order of decreasing reward
        taskList = sorted(passedTask, key= lambda x:x[1], reverse=True)
        for task in failedTask:
            idx = 0
            for j in range(len(taskList)):
                if taskList[j][0] == task[0][0]:
                    idx = j

            taskList.insert(idx+1, task)
        print(taskList)
        return taskList

    else:
        return 0

# Exercise 2.3

print(*"Exercise 2.3")

performance.append(expectedState_Value(choosePolicy('sequential')))
print(" Expected Reward of the Policy: 'Sequential Order' is {}".format(performance[idx_policy["sequential"]]))

#sort the task order following the order of increasing difficulty
performance.append(expectedState_Value(choosePolicy('increasing difficulty')))
print(" Expected Reward of the Policy: 'Increasing Difficulty Order' is {}".format(performance[idx_policy["increasing difficulty"]]))

print()

# Exercise 2.4
print(*"Exercise 2.4")
#sort the task order following the order of increasing reward
performance.append(expectedState_Value(choosePolicy('increasing reward')))
print(" Expected Reward of the Policy: 'Increasing Reward Order' is {}".format(performance[idx_policy["increasing reward"]]))

#sort the task order following the order of decreasing difficulty
performance.append(expectedState_Value(choosePolicy('decreasing difficulty')))
print(" Expected Reward of the Policy: 'Decreasing Difficulty Order' is {}".format(performance[idx_policy["decreasing difficulty"]]))

#sort the task order following the order of decreasing reward
performance.append(expectedState_Value(choosePolicy('decreasing reward')))
print(" Expected Reward of the Policy: 'Decreasing Reward Order' is {}".format(performance[idx_policy["decreasing reward"]]))