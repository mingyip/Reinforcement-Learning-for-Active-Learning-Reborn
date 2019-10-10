
import numpy as np
import keras

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

class emnist_cnn(object):
    def __init__(self, input_shape, nclass):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), 
                            activation='relu', 
                            input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(26, activation='softmax'))

        # initiate RMSprop optimizer
        # opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

        # Let's train the model using Adadelta
        model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy'])

        # Variables
        self.model = model

    def reset_op(self):
        # initiate RMSprop optimizer
        # opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

        # Let's train the model using RMSprop
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy'])