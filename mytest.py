from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.utils import np_utils
from keras import optimizers
from keras.layers.normalization import BatchNormalization
import random

x = np.array(range(100000))
x = x/1000.0
x = x.reshape((-1, 20))
print x.shape
print x
# y = np.array([1 if i > 0.01*len(x) else 0 for i in range(len(x))])
y = np.array([0 for i in range(len(x))])
for j in range(80):
    x[j][random.choice(range(20))] = -4 * random.random()
    y[j] = 1
for j in range(80, 160, 1):
    x[j][random.choice(range(10))] = -2 * random.random()
    x[j][random.choice(range(10, 20, 1))] = -2 * random.random()
    y[j] = 2
for j in range(160, 180, 1):
    x[j][random.choice(range(8))] = -1 * random.random()
    x[j][random.choice(range(8, 12, 1))] = -2 * random.random()
    x[j][random.choice(range(12, 20, 1))] = -1 * random.random()
    y[j] = 3

print np.sum(y)
y = np_utils.to_categorical(y, 4)
print y

# exit()
model = Sequential()
# model.add(Dense(12, activation='tanh', input_shape=(10, )))
# model.add(Dense(2, activation='softmax'))
# model.add(BatchNormalization(input_shape=(10, )))
model.add(Dense(64, input_shape=(20, ), activation='tanh'))
model.add(Dense(32, activation='tanh'))
model.add(Dense(16, activation='tanh'))
model.add(Dense(8, activation='tanh'))
model.add(Dense(4, activation='softmax'))
# model.add(Dense(1))
# model.add(Dense(32, activation='tanh'))
# model.add(Dense(2, activation='softmax'))  # Compile model
# model.add(Dense(128, input_shape=(10, )))
# model.add(Dense(2, activation='softmax'))  # Compile model
sgd = optimizers.SGD(lr=0.005, clipvalue=0.5)
adadelta = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
# adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=adadelta, metrics=['accuracy'])
model.fit(x, y, epochs=15, batch_size=10, validation_split=0.1)
print model.predict(x[50:52])
print y[50:52]
print "#"*20
print model.predict(x[-10:])
print y[-10:]
print "#"*20
print model.predict(x[160:170])
print y[160:170]

