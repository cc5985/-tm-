import numpy as np
import generate_numbers as gn

np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import RMSprop

# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28), y shape (10,000, )

# data pre-processing
# X_train = X_train.reshape(X_train.shape[0], -1) / 255.   # normalize
# X_test = X_test.reshape(X_test.shape[0], -1) / 255.      # normalize
# y_train = np_utils.to_categorical(y_train, num_classes=10)
# y_test = np_utils.to_categorical(y_test, num_classes=10)
total_nums=1000
train_nums=800
divider=2
train_epochs=1000
input_dimension=len(str(total_nums))+1

raw_x=np.linspace(1,total_nums,total_nums,dtype=np.int)
np.random.shuffle(raw_x)
x=gn.generate_seq(raw_x,total_nums)
y=raw_x % divider

print x[:20]
print y[:20]

X_train, Y_train = x[:train_nums], y[:train_nums]
X_test, Y_test = x[train_nums:], y[train_nums:]
X_train=X_train
X_test=X_test

Y_test=np_utils.to_categorical(Y_test, num_classes=divider)
Y_train=np_utils.to_categorical(Y_train, num_classes=divider)

# Another way to build your neural net
model = Sequential([
    Dense(512, input_dim =input_dimension),
    Activation('relu'),
    Dropout(0.2),
    Dense(256),
    Activation('relu'),
    Dense(256),
    Activation('relu'),
    Dense(256),
    Activation('relu'),
    Dense(256),
    Activation('relu'),
    Dense(256),
    Activation('relu'),
    Dense(256),
    Activation('relu'),
    Dense(256),
    Activation('relu'),
    Dense(128),
    Activation('relu'),
    Dense(64),
    Activation('relu'),
    Dense(32),
    Activation('relu'),
    Dense(16),
    Activation('relu'),
    Dense(divider),
    Activation('softmax'),
])

# Another way to define your optimizer
rmsprop = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
# rmsprop = RMSprop()
# We add metrics to get more results you want to see
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')
# Another way to train the model
model.fit(X_train, Y_train, epochs=train_epochs, batch_size=train_nums/10)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, Y_test)

print('test loss: ', loss)
print('test accuracy: '+ str((accuracy*100))+ "%")
