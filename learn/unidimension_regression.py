from keras.models import Sequential
from keras.layers.core import Dense, Activation

model = Sequential()
model.add(Dense(10, 64))
model.add(Activation('tanh'))
model.add(Dense(64, 1))
model.compile(loss='mean_absolute_error', optimizer='rmsprop')

model.fit(X_train, y_train, nb_epoch=20, batch_size=16)
score = model.evaluate(X_test, y_test, batch_size=16)