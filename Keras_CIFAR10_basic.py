# -*- coding: utf-8 -*-

from __future__ import print_function
from matplotlib import pyplot
from scipy.misc import toimage

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import regularizers
from keras import initializers
import keras
import time

#%% Data Preparation
# Load data
t1 = time.time()
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

print('Training set', X_train.shape, y_train.shape)
print('Test set', X_test.shape, y_test.shape)

# Visualize some of them
# Create a grid of 3x3 images
for i in range(0, 9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(toimage(X_train[i]))
# Show the plot
pyplot.show()
X_train.shape[1:]

# Normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# One hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

#%% Model definition
# Create the model
model = Sequential()
# Conv Stack 1
model.add(Conv2D(input_shape=(32, 32, 3), 
                 filters=32, 
                 kernel_size=(3, 3),
                 padding='same', 
                 activation='relu',
                 kernel_initializer=initializers.he_normal(),
                 bias_initializer='zeros'))
model.add(Conv2D(filters=32, 
                 kernel_size=(3, 3),
                 padding='same', 
                 activation='relu',
                 kernel_initializer=initializers.he_normal(),
                 bias_initializer='zeros'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.25))
# Conv Stack 2
model.add(Conv2D(filters=64, 
                 kernel_size=(3, 3),
                 padding='same', 
                 activation='relu',
                 kernel_initializer=initializers.he_normal(),
                 bias_initializer='zeros'))
model.add(Conv2D(filters=64, 
                 kernel_size=(3, 3),
                 padding='same', 
                 activation='relu',
                 kernel_initializer=initializers.he_normal(),
                 bias_initializer='zeros'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.25))
# Conv Stack 3
model.add(Conv2D(filters=128, 
                 kernel_size=(3, 3),
                 padding='same', 
                 activation='relu',
                 kernel_initializer=initializers.he_normal(),
                 bias_initializer='zeros'))
model.add(Conv2D(filters=128, 
                 kernel_size=(3, 3),
                 padding='same', 
                 activation='relu',
                 kernel_initializer=initializers.he_normal(),
                 bias_initializer='zeros'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.25))
#↨ FC stack
model.add(Flatten())
model.add(Dense(units=512, 
                activation='relu',
                kernel_initializer=initializers.he_normal(),
                bias_initializer='zeros',
                kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(units=128, 
                activation='relu',
                kernel_initializer = initializers.he_normal(),
                bias_initializer='zeros',
                kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

#%% Run & Evaluate & Save
# Set parameters and optimizer
epochs = 10
lr=1e-3
decay=0
opt = keras.optimizers.rmsprop(lr=lr, decay=decay)
# Compile model
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
print(model.summary())
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=128)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
# Print time
t2 = time.time()
print("Time: %0.2fs" % (t2 - t1))
# Save
Model_name='Keras_3x2Conv-3Fc_rms-lr%s-decay%s_epoch%s' % (lr,decay,epochs) 
model.save(Model_name)
