import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

# TODO filepath 수정
file_path = os.path.join(os.getcwd(), 'sample.txt')

def read_data_AND_plot_confusion_matrix(file_path):
    with open(file_path,'r') as f:
        lines = f.readlines()

        # variable of pre-parse raw data from filestream
        Title_preparse_list = [index for index, value in enumerate(lines) if 'Title:\n' in value]
        Classes_preparse_list = [index for index, value in enumerate(lines) if 'Classes:\n' in value]
        Confusion_Matrix_preparse_list = [index for index, value in enumerate(lines) if  'Confusion Matrix:\n' in value]



        # post parsing
        Title_list = []
        Confusion_Matrix_list = []
        Classes_list = []
        # iter = total number of set(title, #classes, confusion matrix)
        iteration = len(Title_preparse_list)
        for iter in range(iteration):

            title = lines[Title_preparse_list[iter]+1]
            Title_list.append(title)

            classes_preparse = lines[Classes_preparse_list[iter]+1]
            if iter == iteration-1:
                # Confusion Matrix before parse
                CM_preparse =lines[Confusion_Matrix_preparse_list[iter]+1:]
            else:
                CM_preparse = lines[Confusion_Matrix_preparse_list[iter]+1:Title_preparse_list[iter+1]]


            # build the array of classes
            classes_preparse = classes_preparse.split(' ')
            classes_parsed = []

            for class_preparse in classes_preparse:

                # find where qoutes are and extract strings between those qoutes
                if class_preparse.find("'") != None:
                    indices_of_qoute = [index for index, value in enumerate(class_preparse) if value == "'"]
                    class_parsed = class_preparse[indices_of_qoute[0]+1:indices_of_qoute[-1]]
                    classes_parsed.append(class_parsed)

            Classes_list.append(classes_parsed)
            num_classes = len(classes_parsed)

            # build the numpy array of confusion matrix
            confusion_matrix_elements = []
            CM_preparse = " ".join([str(x) for x in CM_preparse])
            CM_elements = CM_preparse.split(" ")

            for element in CM_elements:

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

            print(confusion_matrix_elements)
            print('?')
            # convert string into int and reshape it as 2-D matrix
            confusion_matrix_flatten = np.asarray([int(x) for x in confusion_matrix_elements])
            class_percentages = ["{0:0.2%}".format(value) for value in confusion_matrix_flatten/sum(confusion_matrix_flatten)]

            heatmap_label = [f"{v1}\n{v2}" for v1, v2 in zip(confusion_matrix_flatten, class_percentages)]
            heatmap_label = np.asarray(heatmap_label).reshape(num_classes,-1)
            confusion_matrix = confusion_matrix_flatten.reshape(num_classes, -1)
            Confusion_Matrix_list.append(confusion_matrix)

            f.close()

            # plot and save confusion matrix as png
            # df_cm = pd.DataFrame(confusion_matrix, index = [i for i in range(num_classes)],
            #                      columns = [i for i in range(num_classes)])
            df_cm = pd.DataFrame(confusion_matrix, index = [i for i in classes_parsed],
                                 columns = [i for i in classes_parsed])
            print([i for i in classes_parsed])
            plt.figure(figsize=(20, 20))
            plt.title(title)
            sn.heatmap(df_cm, annot=heatmap_label, fmt='', cbar=False)
            plt.xlabel('Prediction')
            plt.ylabel('True Label')
            plt.savefig(os.getcwd()+'/figures/'+title+'.png')
            #plt.show()


    return Title_list, Classes_list, Confusion_Matrix_list

def precision_AND_recall_plot(Confusion_Matrix_list, Title_list, Classes_list):

    for iter in range(len(Title_list)):
        title = Title_list[iter]
        print(Title_list[iter])
        confusion_matrix = Confusion_Matrix_list[iter]
        Classes = Classes_list[iter]
        num_classes = len(Classes)

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



        #TODO modify here for the case of 101 classes
        if len(Classes) == 101:
            # list of set(precision, recall, class)
            PRC_list = list(zip(precisions, recalls, Classes))

            # sort by recall
            best_worst_5_PRC = sorted(PRC_list, key=lambda x:x[1])
            best_worst_5_PRC = best_worst_5_PRC[0:5]+best_worst_5_PRC[-5:]

            #recall and precision list
            RP_list = [recall for precision,recall,label in best_worst_5_PRC] + [precision for precision,recall, label in best_worst_5_PRC]
            RP_matrix = np.asarray(RP_list).reshape(2,-1)
            RP_class = [label for precision,recall,label in best_worst_5_PRC]

        # when num_classes is below 101
        else:
            #recall and precision list
            RP_list = recalls + precisions
            RP_matrix = np.asarray(RP_list).reshape(2,-1)
            RP_class = Classes

        print('label'+str(RP_list))
        heatmap_label = ["{0:0.2f}".format(value) for value in RP_list]
        heatmap_label = np.asarray(heatmap_label).reshape(2,-1)

        df_RP_matrix = pd.DataFrame(RP_matrix, columns = [i for i in RP_class], index=["Recall", "Precision"])


        plt.figure(figsize=(20, 5))
        plt.title(title)
        sn.heatmap(df_RP_matrix, annot=heatmap_label, fmt='', cbar=True)
        plt.xlabel('Class')
        plt.savefig(os.getcwd()+'/figures/'+title+'_precision&recall'+'.png')
    return

if __name__ == '__main__':

    (Title_list, Classes_list, Confusion_Matrix_list) = read_data_AND_plot_confusion_matrix(file_path)
    precision_AND_recall_plot( Confusion_Matrix_list=Confusion_Matrix_list, Title_list=Title_list,Classes_list=Classes_list)