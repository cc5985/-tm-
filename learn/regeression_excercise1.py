import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

# create some data
X = np.linspace(1, 1000, 10000, dtype=float)

np.random.shuffle(X)    # randomize the data
Y = np.log(X)

X=X/1000.0
# plot data
plt.scatter(X, Y)
# plt.show()
#
X_train, Y_train = X[:800], Y[:800]
X_test, Y_test = X[800:], Y[800:]
#
#
model = Sequential()
model.add(Dense(output_dim=32, input_dim=1))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='relu'))


# # choose loss function and optimizing method
model.compile(loss='mse', optimizer='sgd')
#
# # training
print('Training -----------')
for step in range(30001):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 1000 == 0:
        print('train cost: ', cost)

# test
print('\nTesting ------------')
cost = model.evaluate(X_test, Y_test, batch_size=1000)
print('test cost:', cost)
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)

# plotting the prediction
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()