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

# load dataset
X_train, Y_train = load_mnist("training")
X_test, Y_test = load_mnist("testing")

# normalize the input data
X_train = X_train / 255.
X_test = X_test / 255.

# reshape dimension to [#data, 784]
X_train = np.reshape(X_train, (np.shape(X_train)[0], -1))
X_test = np.reshape(X_test, (np.shape(X_test)[0], -1))


# one-hot encoding for labels
Y_train = tf.keras.utils.to_categorical(Y_train, 10)
Y_test = tf.keras.utils.to_categorical(Y_test, 10)

num_epoch = 25
num_class = 10


def build_network(X_train, Y_train, X_test, Y_test, regularizer, network_optimizer, num_class):
    """
    
    :param X_train: 
    :param Y_train: 
    :param X_test: 
    :param Y_test: 
    :param regularizer: 
    :param optimizer: 
    :return: 
    """

    model = Sequential()

    # input shape=[batch_size, 784]
    model.add(Dense(units=512, activation='relu', use_bias=True, kernel_regularizer=regularizer, input_dim=(784)))

    # input shape = [batch_size, 512]
    model.add(Dense(units=512, activation='relu', use_bias=True, kernel_regularizer=regularizer))

    # input shape = [batch_size, 512]
    model.add(Dense(units=num_class, activation='softmax', use_bias=True, kernel_regularizer=regularizer))

    optimizer = network_optimizer
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    return model


# train models, evaluate and plot
def eval_and_plot(X_train, Y_train, X_test, Y_test, num_class, validation_rate):
    """

    :param X_train:
    :param Y_train:
    :param X_test:
    :param Y_test:
    :param num_class:
    :param validation_rate:
    :return:
    """
    SGD = build_network(X_train, Y_train, X_test, Y_test, None, tf.keras.optimizers.SGD(), num_class=num_class)
    SGD_nesterov = build_network(X_train, Y_train, X_test, Y_test, None, tf.keras.optimizers.SGD(nesterov=True),
                                 num_class=num_class)
    SGD_nesterov_L1 = build_network(
        X_train, Y_train, X_test, Y_test, tf.keras.regularizers.l1(l=0.001), tf.keras.optimizers.SGD(nesterov=True),
        num_class=num_class)
    SGD_nesterov_L2 = build_network(X_train, Y_train, X_test, Y_test, tf.keras.regularizers.l2(l=0.01),
                                    tf.keras.optimizers.SGD(nesterov=True),
                                    num_class=num_class)

    #history = SGD.fit(x=X_train,y=Y_train,epochs=5, validation_split=validation_rate)
    #print(history.history)

    networks = [SGD, SGD_nesterov, SGD_nesterov_L1, SGD_nesterov_L2]
    name = ['SGD', 'SGD_nesterov', 'SGD_nesterov_L1', 'SGD_nesterov_L2']
    histories = []

    # train and save loss as well as accuracies
    for index, network in enumerate(networks):
        history = network.fit(x=X_train,y=Y_train,epochs=num_epoch, validation_split=validation_rate)

        x_range = range(num_epoch)
        plt.plot(x_range, history.history['val_loss'], label=name[index])


    # plot the figure and save
    plt.legend(['SGD','SGD_nesterov','SGD_nesterov_L1','SGD_nesterov_L2'], loc='best')
    plt.show()
    plt.savefig('./validation_comparision.png')


if __name__ == "__main__":
    eval_and_plot(X_train, Y_train, X_test, Y_test, num_class, validation_rate=0.3)
