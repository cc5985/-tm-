import numpy as np
import generate_numbers as gn
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import RMSprop
import keras
charset=list('''0123456789~`!@#$%^&*()_+QWERTYUIOP{}ASDFGHJKL:"|ZXCVBNM<>?qwertyuiop[]asdfghjkl;'\zxcvbnm,./ ''')
numbers=list('''0123456789''')
# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28), y shape (10,000, )

# data pre-processing
# X_train = X_train.reshape(X_train.shape[0], -1) / 255.   # normalize
# X_test = X_test.reshape(X_test.shape[0], -1) / 255.      # normalize
# y_train = np_utils.to_categorical(y_train, num_classes=10)
# y_test = np_utils.to_categorical(y_test, num_classes=10)
total_nums=1000
train_nums=800
output_dim=2
train_epochs=200
input_dimension=3
input_dim=len(charset)

def set_X_Y(length=10000):
    result={}
    X=[]
    Y=[]
    for cnt in range(0,length):
        choice=np.random.choice(charset)
        if choice in numbers:
            y=1
        else:
            y=0
        choice=ord(choice)-34
        X.append(choice)
        Y.append(y)
    X=np.array(X,dtype=np.int)
    Y=np.array(Y,dtype=np.int)
    result["X"]=X
    result["Y"]=Y
    return result

result=set_X_Y(total_nums)
X=result["X"]
Y=result["Y"]
X=keras.utils.to_categorical(X,input_dim)
Y=keras.utils.to_categorical(Y,output_dim)

X_train, Y_train = X[:train_nums], Y[:train_nums]
X_test, Y_test = X[train_nums:], Y[train_nums:]


# # Another way to build your neural net
model = Sequential([
    Dense(64, input_dim =input_dim),
    Activation('relu'),
    Dense(32),
    Activation('relu'),
    Dense(output_dim),
    Activation('softmax'),
])

print model.summary()
# # Another way to define your optimizer
rmsprop = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
# # rmsprop = RMSprop()
# # We add metrics to get more results you want to see
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
