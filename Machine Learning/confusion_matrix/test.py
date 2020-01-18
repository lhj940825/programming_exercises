import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

## @reference
## https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea


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

    plt.figure(figsize=(10, 10))
    plt.title(title)
    sn.heatmap(df_cm, annot=True, fmt='', cbar=False)
    plt.xlabel('Prediction')
    plt.ylabel('True Label')
    plt.show()

    # group_names = [‘True Neg’,’False Pos’,’False Neg’,’True Pos’]
    # group_counts = [“{0:0.0f}”.format(value) for value in
    # cf_matrix.flatten()]
    # group_percentages = [“{0:.2%}”.format(value) for value in
    # cf_matrix.flatten()/np.sum(cf_matrix)]
    # labels = [f”{v1}\n{v2}\n{v3}” for v1, v2, v3 in
    # zip(group_names,group_counts,group_percentages)]
    # labels = np.asarray(labels).reshape(2,2)
    #




file_path = os.getcwd()+'/sample.txt'
read_data_AND_plot_confusion_matrix(file_path=file_path, num_classes=30, title='Confusion Matrix')