# -*- coding: utf-8 -*-
"""
Created on 2019.12.26
Auther : jun

"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import SGD

from load_mnist import *
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import Callback


class LR_Callback(Callback):
    def __init__(self, test_data, label, loss_log):
        self.test_data = test_data
        self.label = label
        self.loss_log = loss_log

    def on_batch_end(self, batch, logs={}):
        x = self.test_data
        y = self.label
        loss, acc = self.model.evaluate(x, y, verbose=0)
        self.loss_log.append(loss)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

class Train_Err_Callback2(Callback):
    def __init__(self, loss_log):
        self.loss_log = loss_log

    def on_batch_end(self, batch, logs={}):

        # store training errors
        self.loss_log.append(logs['loss'])


def build_network(regularizer, network_optimizer, num_class):
    """

    :param regularizer:
    :param optimizer:
    :return:
    """

    model = Sequential()
    model.add(tf.keras.layers.Conv2D(filters=64, strides=1, padding='valid', kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(tf.keras.layers.Conv2D(filters=128, strides=1, padding='valid', kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(tf.keras.layers.Flatten())
    model.add(Dense(input_dim=(1600),units=num_class,activation='softmax'))
    print(model.summary())
    
    optimizer = network_optimizer
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    return model

def build_network_with_regularizer(regularizer, network_optimizer, num_class):
    """

    :param regularizer:
    :param optimizer:
    :return:
    """

    model = Sequential()
    model.add(tf.keras.layers.Conv2D(filters=64, strides=1, padding='valid', kernel_size=(3, 3), activation='relu', input_shape=(28,28,1),kernel_regularizer = regularizer))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(filters=128, strides=1, padding='valid', kernel_size=(3, 3), activation='relu',kernel_regularizer = regularizer))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(Dense(input_dim=(1600),units=num_class,activation='softmax',kernel_regularizer = regularizer))
    print(model.summary())

    optimizer = network_optimizer
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    return model


def learning_rates_comparision(X_train,Y_train,X_test,Y_test,num_class, num_epoch):

    #history = conv_network.fit(x=X_train,y=Y_train, batch_size=512,epochs=5, validation_split=0.3)

    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    learning_rates_name = ['0.1','0.01','0.001','0.0001']
    for idx, learning_rate in enumerate(learning_rates):
        loss_log = []
        conv_network= build_network(None, tf.keras.optimizers.SGD(lr=learning_rate), num_class)
        history = conv_network.fit(x=X_train, y=Y_train, batch_size=3000, epochs=num_epoch, validation_split=0.3, callbacks = [LR_Callback(X_test, Y_test, loss_log)])
        plt.plot(loss_log, label=learning_rates_name[idx])

    plt.legend(loc='best')
    plt.xlabel('iteration')
    plt.ylabel('classification error')
    plt.title('learning rate comparision')
    plt.savefig('./learning_rate comparision.png')
    plt.show()


def train_and_plot_optimal_network(X_train, Y_train, X_test,Y_test, num_class, num_epoch):
    # build network with the optimal learning rate
    best_network = build_network('None', tf.keras.optimizers.Adam(lr=0.01), num_class)

    loss_log = []
    history = best_network.fit(x=X_train, y=Y_train, batch_size=3000, epochs=num_epoch, validation_split=0.3, callbacks = [Train_Err_Callback2(loss_log)])
    history = best_network.evaluate(X_test,Y_test)

    fig = plt.figure()
    plt.plot(loss_log)
    plt.title('Final Training Loss over interation')
    plt.xlabel('Iternation')
    plt.ylabel('Loss')
    plt.savefig('./training_loss_figure.png')
    plt.show()

    print("Final Testing accuracy: "+ str(history[1]))

if __name__ == '__main__':
    # load dataset
    X_train, Y_train = load_mnist("training")
    X_test, Y_test = load_mnist("testing")

    # normalize the input data
    X_train = X_train / 255.
    X_test = X_test / 255.

    # reshape image
    img_rows = 28
    img_cols = 28
    X_train = np.reshape(X_train, [np.shape(X_train)[0],img_rows, img_cols,1 ])
    X_test = np.reshape(X_test, [np.shape(X_test)[0],img_rows, img_cols,1 ])

    # one-hot encoding for labels
    Y_train = tf.keras.utils.to_categorical(Y_train, 10)
    Y_test = tf.keras.utils.to_categorical(Y_test, 10)

    num_epoch = 1
    num_class = 10

    ## Task 1: In order to find the optimal learning rate, activate the function below,
    learning_rates_comparision(X_train,Y_train,X_test,Y_test,num_class,num_epoch)


    ## Task 2: Plot the training loss over the iterations and report the final test accuracy
    train_and_plot_optimal_network(X_train,Y_train, X_test, Y_test, num_class, num_epoch)

    # Task 3: Add regularization to the network in the form of dropout layers and weight decay.
    best_network = build_network_with_regularizer(tf.keras.regularizers.l2(l=0.001), tf.keras.optimizers.Adam(lr=0.01), num_class)
    history = best_network.fit(x=X_train, y=Y_train, batch_size=3000, epochs=num_epoch, validation_split=0.3)
    history = best_network.evaluate(X_test,Y_test)
    print("Final Testing accuracy with regularizers: "+ str(history[1]))








