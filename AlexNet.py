import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, BatchNormalization
from keras.utils import np_utils

# Import CIFAR100 small classification dataset

from keras.datasets import cifar100

(Xtrain, ytrain), (X_test, y_test) = cifar100.load_data(label_mode = 'fine')
X_train, X_val, y_train, y_val = train_test_split(Xtrain, ytrain,
                                                  test_size=0.20,
                                                  random_state=42)

# Defining early parameters

N, w, h, c = Xtrain.shape
nb_classes = 100

# Shuffle training data
perm = np.arange(len(X_train))
np.random.shuffle(perm)
X_train = X_train[perm]
y_train = y_train[perm]

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_val.shape[0], 'validation samples')
print(X_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
Y_val = np_utils.to_categorical(y_val, nb_classes)

# Normalize input data (NOTE: - Original AlexNet paper didn't implement input Normalization, rather used layer normalization)

X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_val /= 255
X_test /= 255


# Original AlexNet uses 224x224x3 RGB Images from ImageNet dataset, we'll be using 32x32x3 from CIFAR100 dataset

class AlexNet:
    """
        Sci-kit type based model for implementing AlexNet*
    """

    def __init__(self, n_classes, kernel_sizes, fc_layer_sizes, *input_shape):
        (self.N, self.w, self.h, self.c) = input_shape
        self.n_class = n_classes
        self.kernel_sizes = kernel_sizes
        self.fc_layer_sizes = fc_layer_sizes
        self.model = Sequential()
        assert len(kernel_sizes) == 5 and type(kernel_sizes) == list
        assert len(fc_layer_sizes) == 2 and type(fc_layer_sizes) == list

    def initialize_model(self):
        self.model.add(Convolution2D(filters=32, input_shape=(self.w, self.h, self.c), kernel_size=self.kernel_sizes[0],
                                     strides=(1, 1), padding='same', bias_initializer='zeros'))
        self.model.add(BatchNormalization(axis=1))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dropout(0.3))

        self.model.add(Convolution2D(filters=64, kernel_size=self.kernel_sizes[1], strides=(1, 1),
                                     padding='same', bias_initializer='zeros'))
        self.model.add(BatchNormalization(axis=1))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dropout(0.3))

        # self.model.add(Convolution2D(filters=128, kernel_size=self.kernel_sizes[2], strides=(1, 1),
        #                              padding='same', bias_initializer='zeros'))
        # self.model.add(Activation('relu'))
        # self.model.add(Convolution2D(filters=256, kernel_size=self.kernel_sizes[3], strides=(1, 1),
        #                              padding='same', bias_initializer='zeros'))
        # self.model.add(Activation('relu'))
        # self.model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
        # self.model.add(Convolution2D(filters=256, kernel_size=self.kernel_sizes[3], strides=(1, 1), activation='relu',
        #                              padding='same', bias_initializer='zeros'))
        # self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(self.fc_layer_sizes[0], activation='relu'))
        self.model.add(Dropout(rate=0.3))
        self.model.add(Dense(self.fc_layer_sizes[1], activation='relu'))
        self.model.add(Dropout(rate=0.3))
        self.model.add(Dense(self.n_class, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

    def train_model(self, X, y, X_val, y_val, batch_size, n_epochs, verbose=1):
        self.model.fit(X, y, batch_size=batch_size, epochs=n_epochs, verbose=verbose, validation_data=(X_val, y_val),
              shuffle=True)

    def eval_model(self, Xtest, ytest, verbose):
        score = self.model.evaluate(Xtest, ytest, verbose=verbose)
        return score


# Check score with CIFAR dataset

alexNet = AlexNet(100, [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)], [4096, 4096], *Xtrain.shape)
alexNet.initialize_model()
alexNet.train_model(X_train, Y_train, X_val, Y_val, 128, 100, 1)
score = alexNet.eval_model(X_test, Y_test, verbose=1)

print(score)