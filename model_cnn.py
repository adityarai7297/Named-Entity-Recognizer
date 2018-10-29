import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import pickle


with open('Xtrain.pkl', 'rb') as f:
   Xtrain = pickle.load(f)
Xtrain=Xtrain.reshape(51678, 15, 26,1)

with open('Ytrain.pkl', 'rb') as f1:
   Ytrain = pickle.load(f1)
#print(Xtrain.shape)

batch_size = 150
num_classes = 1
epochs = 50


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(15,26,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(Xtrain, Ytrain,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)


model.save('trained_model.h5')  