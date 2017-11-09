# coding: utf-8

import numpy as np
from scipy.sparse import issparse
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D, Conv1D
from keras.layers.convolutional import MaxPooling2D, MaxPooling1D
from keras.utils import np_utils
import tensorflow as tf


def load_data(path='mnist.npz'):
    """Loads the MNIST dataset.

    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)

# 加载数据
(X_train, y_train), (X_test, y_test) = load_data()

print "X_train shape = ", X_train.shape
print y_train[0]

# print X_train[0]
# exit()


def transform_data(X_train, y_train, X_test, y_test):
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
    y_train = np_utils.to_categorical(y_train)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]
    return X_train, y_train, X_test, y_test, num_classes


# num_classes = np_utils.to_categorical(y_test).shape[1]
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)


# 简单的CNN模型
def baseline_model():
    # create model
    model = Sequential()
    # 卷积层
    model.add(Conv2D(32, (3, 3), padding='valid', input_shape=(28, 28, 1), activation='sigmoid'))  # 池化层
    # model.add(Conv2D(32, (3, 3), padding='valid', input_shape=(28, 28), activation='relu'))  # 池化层
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 卷积层
    model.add(Conv2D(20, (3, 3), padding='valid', input_shape=(28, 28, 1), activation='tanh'))  # 池化层
    # model.add(Conv2D(32, (3, 3), padding='valid', input_shape=(28, 28), activation='relu'))  # 池化层
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 卷积
    model.add(Conv2D(15, (3, 3), padding='valid', activation='tanh'))  # 池化
    # model.add(Conv2D(12, (3, 3), padding='valid', activation='relu'))  # 池化
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 全连接，然后输出
    model.add(Flatten())
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(num_classes, activation='softmax'))  # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def baseline_model1():
    # create model
    model = Sequential()
    model.add(Conv1D(32, 5, border_mode='same', input_shape=(28, 28), activation='relu'))
    model.add(MaxPooling1D())
    model.add(Conv1D(20, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling1D())
    model.add(Conv1D(12, 3, border_mode='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))  # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def prec_ets(n_trees, X_train, y_train, X_test, y_test, random_state=None):
    """
    ExtraTrees
    """
    from sklearn.ensemble import ExtraTreesClassifier
    if not issparse(X_train):
        X_train = X_train.reshape((X_train.shape[0], -1))
    if not issparse(X_test):
        X_test = X_test.reshape((X_test.shape[0], -1))
    clf = ExtraTreesClassifier(n_estimators=n_trees, max_depth=None, n_jobs=-1, verbose=1, random_state=random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print np.argmax(y_pred, axis=1)
    prec = float(np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1))) / len(y_test)
    return clf, y_pred, prec


def prec_rf(n_trees, X_train, y_train, X_test, y_test, random_state=None):
    """
    ExtraTrees
    """
    from sklearn.ensemble import RandomForestClassifier
    if not issparse(X_train):
        X_train = X_train.reshape((X_train.shape[0], -1))
    if not issparse(X_test):
        X_test = X_test.reshape((X_test.shape[0], -1))
    clf = RandomForestClassifier(n_estimators=n_trees, max_depth=None, n_jobs=-1, verbose=1, random_state=random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    prec = float(np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1))) / len(y_test)
    return clf, y_pred, prec


# print "prec_ets", prec_ets(200, X_train, y_train, X_test, y_test, 0)[2]
# print "prec_rf", prec_rf(200, X_train, y_train, X_test, y_test, 0)[2]

# build the model
X_train, y_train, X_test, y_test, num_classes = transform_data(X_train, y_train, X_test, y_test)
model = baseline_model()

print X_train.shape
# model = baseline_model1()

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=128, verbose=2)



