import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Activation, Flatten, Dense
from keras.utils import np_utils
from sklearn.utils import shuffle

## Dataset -- CIFAR100 object classification dataset

# Import CIFAR100 small classification dataset

from keras.datasets import cifar100

(Xtrain, ytrain), (X_test, y_test) = cifar100.load_data(label_mode='fine')
X_train, X_val, y_train, y_val = train_test_split(Xtrain, ytrain,
                                                  test_size=0.20,
                                                  random_state=42)  # random state for reproducibility

### Normalization of input, shuffle and encoding of one-hot vectors of labels

# Normalization

X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_val /= 255
X_test /= 255

### Model parameters

N, w, h, c = X_train.shape
nb_classes = 100

# Shuffle training data
X_train, y_train = shuffle(X_train, y_train)

# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
Y_val = np_utils.to_categorical(y_val, nb_classes)


## Main class

class VGGNet:
    model = Sequential()

    def __init__(self, n_classes, kernel_size, *input_shape):
        print(input_shape)
        (self.N, self.w, self.h, self.c) = input_shape
        self.n_class = n_classes
        self.kernel_size = kernel_size

    def initialize(self):
        self.model.add(Convolution2D(filters=64, input_shape=(self.w, self.h, self.c), kernel_size=self.kernel_size,
                                     bias_initializer='zeros', strides=(1, 1), padding='same',
                                     data_format='channels_last', activation='relu'))
        self.model.add(Convolution2D(filters=64, kernel_size=self.kernel_size, bias_initializer='zeros', strides=(1, 1),
                                     padding='same', data_format='channels_last', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

        self.model.add(
            Convolution2D(filters=128, kernel_size=self.kernel_size, bias_initializer='zeros', strides=(1, 1),
                          padding='same', data_format='channels_last', activation='relu'))
        self.model.add(
            Convolution2D(filters=128, kernel_size=self.kernel_size, bias_initializer='zeros', strides=(1, 1),
                          padding='same', data_format='channels_last', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

        self.model.add(
            Convolution2D(filters=256, kernel_size=self.kernel_size, bias_initializer='zeros', strides=(1, 1),
                          padding='same', data_format='channels_last', activation='relu'))
        self.model.add(
            Convolution2D(filters=256, kernel_size=self.kernel_size, bias_initializer='zeros', strides=(1, 1),
                          padding='same', data_format='channels_last', activation='relu'))
        self.model.add(
            Convolution2D(filters=256, kernel_size=self.kernel_size, bias_initializer='zeros', strides=(1, 1),
                          padding='same', data_format='channels_last', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

        self.model.add(
            Convolution2D(filters=512, kernel_size=self.kernel_size, bias_initializer='zeros', strides=(1, 1),
                          padding='same', data_format='channels_last', activation='relu'))
        self.model.add(
            Convolution2D(filters=512, kernel_size=self.kernel_size, bias_initializer='zeros', strides=(1, 1),
                          padding='same', data_format='channels_last', activation='relu'))
        self.model.add(
            Convolution2D(filters=512, kernel_size=self.kernel_size, bias_initializer='zeros', strides=(1, 1),
                          padding='same', data_format='channels_last', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

        self.model.add(
            Convolution2D(filters=512, kernel_size=self.kernel_size, bias_initializer='zeros', strides=(1, 1),
                          padding='same', data_format='channels_last', activation='relu'))
        self.model.add(
            Convolution2D(filters=512, kernel_size=self.kernel_size, bias_initializer='zeros', strides=(1, 1),
                          padding='same', data_format='channels_last', activation='relu'))
        self.model.add(
            Convolution2D(filters=512, kernel_size=self.kernel_size, bias_initializer='zeros', strides=(1, 1),
                          padding='same', data_format='channels_last', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

        self.model.add(Flatten(data_format='channels_last'))
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dense(100, activation='softmax'))

        self.model.summary()

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

    def train_model(self, X, y, X_val, y_val, batch_size, n_epochs, verbose=1):
        self.model.fit(X, y, batch_size=batch_size, epochs=n_epochs, verbose=verbose, validation_data=(X_val, y_val),
                       shuffle=True)

    def eval_model(self, Xtest, ytest, verbose):
        score = self.model.evaluate(Xtest, ytest, verbose=verbose)
        return score

    ## Evalutation of model


# Check score with CIFAR dataset

VGGNet = VGGNet(100, (3, 3), *X_train.shape)
VGGNet.initialize()
VGGNet.train_model(X_train, Y_train, X_val, Y_val, 128, 10, 1)
score = VGGNet.eval_model(X_test, Y_test, verbose=1)

print(score)