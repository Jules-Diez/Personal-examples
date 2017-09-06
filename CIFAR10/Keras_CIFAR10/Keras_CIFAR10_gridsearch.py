# -*- coding: utf-8 -*-

import time
import numpy as np
import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import regularizers, initializers
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

#%% Data Preparation
t1 = time.time()
num_classes = 10
# Load data
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
print('Training set', X_train.shape, y_train.shape)
print('Test set', X_test.shape, y_test.shape)

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
def create_model(conv_filters, conv_stack,
                 dense_layer_sizes,
                 kernel_size,
                 optimizer='rmsprop', init='he_normal'):
# Create the model
    model = Sequential()
    
    # First stack of Conv - separate because of the input_shape shape
    model.add(Conv2D(input_shape=(32, 32, 3), 
                 filters=conv_filters[0], 
                 kernel_size=kernel_size,
                 padding='same',
                 activation='relu',
                 kernel_initializer=init,
                 bias_initializer='zeros'))
    for stack in range(conv_stack-1) :
       model.add(Conv2D(filters=conv_filters[0], 
                     kernel_size=kernel_size,
                     padding='same',
                     activation='relu',
                     kernel_initializer=init,
                     bias_initializer='zeros')) 
       model.add(Dropout(0.25))
    # CONV layers
    for filters in conv_filters[1:]:
       for stack in range(conv_stack):
           model.add(Conv2D(filters=filters, 
                             kernel_size=kernel_size,
                             padding='same',
                             activation='relu',
                             kernel_initializer=init,
                             bias_initializer='zeros'))
       model.add(MaxPooling2D(pool_size=(2, 2), strides=2))       
       model.add(Dropout(0.25))
    # FC layers
    model.add(Flatten())
    for layer_size in dense_layer_sizes:
    
        model.add(Dense(units=layer_size, 
                        activation='relu',
                        kernel_initializer=init,
                        bias_initializer='zeros',
                        kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.5))
    #model.add(keras.layers.normalization.BatchNormalization(axis=-1))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())
    return model

#%% Params & Grid definition
optimizers = ['rmsprop']
init = ['he_normal']
epochs = [10]
batches = [64]
cv_stack = [2]
kl_size = [(3, 3)]
#cv_filters =[[32,64],[32,64,128]]
cv_filters =[[32,64,128]]
#dense_size_candidates = [[128], [256] ,[512], [256, 128], [512,128]]
dense_size_candidates = [[512,128]]
#dense_size_candidates = [[128],[256]]
param_grid = dict(optimizer=optimizers,
                  epochs=epochs, 
                  batch_size=batches, 
                  init=init,
                  conv_filters=cv_filters,
                  conv_stack=cv_stack,
                  kernel_size=kl_size,
                  dense_layer_sizes=dense_size_candidates)
#%% Model and Grid creation & Running
model = KerasClassifier(build_fn=create_model, verbose=2)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(X_train, y_train)

#%% Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))

t2 = time.time()
print("Time: %0.2fs" % (t2 - t1))