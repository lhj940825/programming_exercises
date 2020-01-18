# Hojun Lim, 18.01.2020


import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

## @reference
# https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
# https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-precision-and-recall-9250280bddc2

class_10 = ['baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets', 'bibimbap',
            'bread_pudding', 'breakfast_burrito', 'bruschetta']

class_30 = ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets',
            'bibimbap', 'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad',
            'carrot_cake', 'ceviche', 'cheese_plate', 'cheesecake', 'chicken_curry', 'chicken_quesadilla',
            'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich',
            'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes']


cwd = os.path.join(os.getcwd(), "confusion_matrix/")

def read_data_AND_plot_confusion_matrix(file_path, num_classes, title ):

    #TODO made below as function
    # read data and save it in 2D array
    with open(file_path,'r') as f:

        # save all strings in file in the variable 'texts'
        texts = ''
        for line in f.readlines():
            texts += line


        confusion_matrix_elements = []
        texts = texts.split(" ")

        for element in texts:

            # get rid of ']' and extract only the number then append it to the list of the confusion matrix elements
            if ']' in element:
                cm_element = element[0:element.find(']')]

                if cm_element.isdigit():
                    confusion_matrix_elements.append(cm_element)

            # get rid of '\n' and extract only the number then append it to the list of the confusion matrix elements
            elif '\n' in element:
                cm_element = element[0:element.find('\n')]

                if cm_element.isdigit():
                  confusion_matrix_elements.append(cm_element)


            else:
                if element.isdigit():
                    confusion_matrix_elements.append(element)

        # convert string into int and reshape it as 2-D matrix
        confusion_matrix_flatten = np.asarray([int(x) for x in confusion_matrix_elements])
        class_percentages = ["{0:0.2%}".format(value) for value in confusion_matrix_flatten/sum(confusion_matrix_flatten)]

        heatmap_label = [f"{v1}\n{v2}" for v1, v2 in zip(confusion_matrix_flatten, class_percentages)]
        heatmap_label = np.asarray(heatmap_label).reshape(num_classes,-1)

        confusion_matrix = confusion_matrix_flatten.reshape(num_classes, -1)

    f.close()



    # plot confusion matrix
    df_cm = pd.DataFrame(confusion_matrix, index = [i for i in range(num_classes)],
                          columns = [i for i in range(num_classes)])

    plt.figure(figsize=(20, 20))
    plt.title(title)
    sn.heatmap(df_cm, annot=heatmap_label, fmt='', cbar=False)
    plt.xlabel('Prediction')
    plt.ylabel('True Label')
    plt.savefig(os.getcwd()+'/confusion_matrix/figures/'+title+'.png')
    #plt.show()

    return confusion_matrix
def precision_AND_recall_plot(confusion_matrix, title, num_classes):
    # list for storing calculated precisions and recalls of the input confusion matrix
    precisions = []
    recalls = []

    # shape of confusion matrix
    cm_shape = np.shape(confusion_matrix)

    # recall = True Positive / (True Positive + False Negative)
    for i in range(0, cm_shape[0]):
        FN = 0
        for j in range(0, cm_shape[1]):
            if i == j:
                TP = confusion_matrix[i][j]

            else:
                FN += confusion_matrix[i][j]
        recall = TP/(TP+FN)
        recalls.append(recall)

    # precision = True Positive / (True Positive + False Positive)
    for i in range(0, cm_shape[0]):
        FP = 0
        for j in range(0, cm_shape[1]):
            if i == j:
                TP = confusion_matrix[j][i]

            else:
                FP += confusion_matrix[j][i]

        #TODO update 필요 여기는 임시방편
        if TP+FP == 0:
            precision = 0
        else:
            precision = TP/(TP+FP)
        precisions.append(precision)
    
    #recall and precision list
    RP_list = recalls + precisions
    RP_matrix = np.asarray(RP_list).reshape(2,-1)

    heatmap_label = ["{0:0.2f}".format(value) for value in RP_list]
    heatmap_label = np.asarray(heatmap_label).reshape(2,-1)

    df_RP_matrix = pd.DataFrame(RP_matrix, columns = [i for i in range(num_classes)], index=["Recall", "Precision"])


    plt.figure(figsize=(20, 5))
    plt.title(title)
    sn.heatmap(df_RP_matrix, annot=heatmap_label, fmt='', cbar=True)
    plt.xlabel('Class')

    plt.savefig(os.getcwd()+'/confusion_matrix/figures/'+title+'.png')
    return

# TODO file path confusion matrix 별로 지정
# TODO Title 수정

file_path = os.getcwd()+'/confusion_matrix/sample.txt'
confusion_matrix = read_data_AND_plot_confusion_matrix(file_path=file_path, num_classes=30, title='Confusion Matrix')
precision_AND_recall_plot(confusion_matrix=confusion_matrix, title="Recall AND Precision", num_classes=30)