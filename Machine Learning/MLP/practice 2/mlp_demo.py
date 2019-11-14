"""
Created on 14.11.2019

@author: Jun
"""
from load_ORL_faces import load_DataSet
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import SGD

# number of classes to distinguish
num_class = 20

# load data sets
X_train, Y_train = load_DataSet(dataType='training')
X_test, Y_test = load_DataSet(dataType='testing')

# normalize the input data
X_test = X_test/255.
X_train = X_train/255.

# conduct the one-hot encoding for labels
Y_train = tf.keras.utils.to_categorical(Y_train, num_class)
Y_test = tf.keras.utils.to_categorical(Y_test, num_class)

# build a model with two Dense(fully connected layer)
model = Sequential()
model.add(Dense(units=60, activation='relu',input_shape=(10304,)))
#model.add(Dense(units=30, activation='sigmoid',input_shape=(60,)))
model.add(Dense(units=num_class, activation='softmax',input_shape=(30,)))
model.summary()

# apply stochastic gradient descent
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)

model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, Y_train, batch_size=120, epochs=20, verbose=2, validation_split=.1)

loss, accuracy  = model.evaluate(X_test, Y_test, verbose=2)

print(f'Classification accuracy: {accuracy:.3}')

# plot the accuracy graph
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('iterations/epochs')
plt.savefig('./loss_graph.png')
plt.show()
