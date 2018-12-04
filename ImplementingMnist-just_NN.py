import keras
import numpy as np
from keras.datasets import mnist
from matplotlib import pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential , load_model
from keras.layers import Dense, Dropout, Activation, Flatten,Input
from keras.layers import Convolution2D, MaxPooling2D



# Data preprocess
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255



'''
if you want to plot the first number
plt.imshow(x_train[0])
plt.show()
'''

# The shape of Y is not correct as its not distinct label is just number ex: 7 not [0,0,0,0,0,0,1,0..]
# ex: print(y_train.shape)
y_train = np_utils.to_categorical(y_train[:],num_classes = 10)
y_test = np_utils.to_categorical(y_test[:],num_classes = 10)

# data pre_processing end
'''
defining the compuational graph (feed forward so it must be sequintal) :
input layer -> hidden state -> hidden state ->out 
'''
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)

model = Sequential()

print(x_train.shape)
#input = Input(shape=x_train.shape)
#model.add(Flatten(x_train))
model.add(Dense(128,activation='relu',input_shape=(1,28,28)))
model.add(Dense(128,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Flatten())
model.add(Dense(10,activation='softmax'))

model.compile( optimizer = 'Adam',
loss = 'categorical_crossentropy',
metrics=['accuracy'])

model.fit(x=x_train,y=y_train,epochs=5,shuffle=True)
score = model.evaluate(x_test, y_test)
model.save("Practice_models")
print(score)

# test the prediction and the real value of test data 
'''
new_model = load_model('Practice_models')
prediction = new_model.predict([x_test[:]])
print("prediction")
print(np.argmax(prediction[5]))

x_test = x_test.reshape((x_test.shape[0],28,28))
plt.imshow(x_test[5])
plt.show()
'''


