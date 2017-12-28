import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt

sample_num=1000
training_times=1000
# create some data
X = np.linspace(-10, 10, sample_num, dtype=np.float64)
np.random.shuffle(X)    # randomize the data

'''
constant function
Y = np.zeros(shape=sample_num,dtype=np.float)

power function: failed.....
Y = X**2+ 2*X+1

Y = np.power(2,X)

Y=np.log10(X)

Y=np.sin(X)

Y=np.arcsin(X)
'''

Y = np.sin(X) + np.log10(np.abs(X)) + X**2+ 2*X+1

# plot data
plt.scatter(X, Y)
# plt.show()

train_sample_upperbound=int(0.8*sample_num)
X_train, Y_train = X[:train_sample_upperbound], Y[:train_sample_upperbound]
X_test, Y_test = X[train_sample_upperbound:], Y[train_sample_upperbound:]

#part3: create models, with 1hidden layers
model = Sequential()
model.add(Dense(1024,input_dim=1))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(256))
# model.add(Activation('relu'))
# model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(1))
model.compile(loss='mean_absolute_error', optimizer='rmsprop')

model.fit(X_train, Y_train, nb_epoch=training_times, batch_size=1000)
score = model.evaluate(X_test, Y_test, batch_size=16)


#
# # training
# print('Training -----------')
# for step in range(training_times):
#     cost = model.train_on_batch(X_train, Y_train)
#     if step % (training_times/10) == 0:
#         print('train cost: ', cost)
#
# """
# Training -----------
# train cost:  4.111329555511475
# train cost:  0.08777070790529251
# train cost:  0.007415373809635639
# train cost:  0.003544030711054802
# """

# test
print('\nTesting ------------')
cost = model.evaluate(X_test, Y_test, batch_size=40)
print('test cost:', cost)
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)

"""
Testing ------------
40/40 [==============================] - 0s
test cost: 0.004269329831
Weights= [[ 0.54246825]]
biases= [ 2.00056005]
"""

# plotting the prediction
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_pred)
# plt.plot(X_test, Y_pred)
plt.show()