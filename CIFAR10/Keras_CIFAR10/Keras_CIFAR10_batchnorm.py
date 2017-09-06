# -*- coding: utf-8 -*-
from __future__ import print_function
import keras
import pandas
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils, plot_model
from keras import regularizers, initializers
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, CSVLogger, LearningRateScheduler
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import SGD

#%% Define plot function
def plot_history(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

#%% Data preparation
# Load data
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
print('Training set', X_train.shape, y_train.shape)
print('Test set', X_test.shape, y_test.shape)

# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

#%% Architecture Parameters 
filter_1 = 50
filter_2 = 100
fc1_size = 250
fc2_size = 75
dropout_conv = 0.3
dropout_fc = 0.5
weight_decay_conv = 0.001
weight_decay_fc = 0.01
kernel_init = initializers.he_normal()
batch_normalization = False

#%% Model Parameters
epochs = 20
lr = 0.01
momentum = 0.9
factor_red = 0.2
patience_red = 25
opt = SGD(lr=lr, momentum=momentum, nesterov=True)
#opt = keras.optimizers.rmsprop(lr=lr, decay=1e-6)
#opt = keras.optimizers.adam(lr=lr)
Model_name = '2Stack_3CONV_3FC_'
Model_optim = 'sgd_lr%smom%sepoch%s_' % (lr,momentum,epochs) 
Model_reduclr = 'fc%spat%s_' % (factor_red,patience_red)
Full_name = Model_name+Model_optim+Model_reduclr

#%% Model definition
# Create the model
model = Sequential()
# CONV STACK 1
model.add(Conv2D(input_shape=(32, 32, 3), 
                 filters=filter_1, 
                 kernel_size=(3, 3),
                 padding='same',
                 kernel_regularizer=regularizers.l2(weight_decay_conv),
                 kernel_initializer=kernel_init,
                 bias_initializer='zeros'))
if batch_normalization:
    model.add(keras.layers.normalization.BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(Conv2D(filters=filter_1, 
                 kernel_size=(3, 3),
                 padding='same',
                 kernel_regularizer=regularizers.l2(weight_decay_conv),
                 kernel_initializer=kernel_init,
                 bias_initializer='zeros'))
if batch_normalization:
    model.add(keras.layers.normalization.BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(Conv2D(filters=filter_1, 
                 kernel_size=(3, 3),
                 padding='same',
                 kernel_regularizer=regularizers.l2(weight_decay_conv),
                 kernel_initializer=kernel_init,
                 bias_initializer='zeros'))
if batch_normalization:
    model.add(keras.layers.normalization.BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
model.add(Dropout(dropout_conv))

# CONV STACK 2
model.add(Conv2D(filters=filter_2, 
                 kernel_size=(3, 3),
                 padding='same',
                 kernel_regularizer=regularizers.l2(weight_decay_conv),
                 kernel_initializer=kernel_init,
                 bias_initializer='zeros'))
if batch_normalization:
    model.add(keras.layers.normalization.BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(Conv2D(filters=filter_2, 
                 kernel_size=(3, 3),
                 padding='same',
                 kernel_regularizer=regularizers.l2(weight_decay_conv),
                 kernel_initializer=kernel_init,
                 bias_initializer='zeros'))
if batch_normalization:
    model.add(keras.layers.normalization.BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(Conv2D(filters=filter_2, 
                 kernel_size=(3, 3),
                 padding='same',
                 kernel_regularizer=regularizers.l2(weight_decay_conv),
                 kernel_initializer=kernel_init,
                 bias_initializer='zeros'))
if batch_normalization:
    model.add(keras.layers.normalization.BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
model.add(Dropout(dropout_conv))

# FC STACK 1
model.add(Flatten())
model.add(Dense(units=fc1_size, 
                kernel_initializer=kernel_init,
                bias_initializer='zeros',
                kernel_regularizer=regularizers.l2(weight_decay_fc)))
model.add(keras.layers.normalization.BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Dropout(dropout_fc))
model.add(Dense(units=fc2_size, 
                kernel_initializer = kernel_init,
                bias_initializer='zeros',
                kernel_regularizer=regularizers.l2(weight_decay_fc)))
model.add(keras.layers.normalization.BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Dropout(dropout_fc))
model.add(Dense(num_classes, activation='softmax'))

## CONV Final stack
#model.add(Conv2D(filters=filter_2, 
#                 kernel_size=(3, 3),
#                 padding='same',
#                 kernel_regularizer=regularizers.l2(weight_decay_conv),
#                 kernel_initializer=kernel_init,
#                 bias_initializer='zeros'))
##model.add(keras.layers.normalization.BatchNormalization(axis=1))
#model.add(Activation('relu'))
#model.add(Conv2D(filters=filter_2, 
#                 kernel_size=(1, 1),
#                 padding='same',
#                 kernel_regularizer=regularizers.l2(weight_decay_conv),
#                 kernel_initializer=kernel_init,
#                 bias_initializer='zeros'))
##model.add(keras.layers.normalization.BatchNormalization(axis=1))
#model.add(Activation('relu'))
#model.add(Conv2D(filters=num_classes, 
#                 kernel_size=(1, 1),
#                 padding='same',
#                 kernel_regularizer=regularizers.l2(weight_decay_conv),
#                 kernel_initializer=kernel_init,
#                 bias_initializer='zeros'))
##model.add(keras.layers.normalization.BatchNormalization(axis=1))
#model.add(GlobalAveragePooling2D())
#model.add(Activation('softmax'))

#%% Callbacks
# Reduce LR
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=factor_red, patience=patience_red, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=1e-5)
# learning rate schedule
def step_decay(epoch):
    initial_lrate = lr
    drop = 0.2
    drop_time = [70,100,130]
    if epoch < drop_time[0]:
        lrate = initial_lrate
    elif epoch < drop_time[1]:
        lrate = initial_lrate*drop
    elif epoch < drop_time[2]:
        lrate = initial_lrate*drop*drop
    else:
        lrate = initial_lrate*drop*drop
    return lrate
LRS = LearningRateScheduler(step_decay)
# CVSlog
csvlog_filepath = Full_name+"csvlog.csv"
cvslog = CSVLogger(csvlog_filepath, separator=',', append=False)
# Checkpoint
checkpoint_filepath = Full_name+"weights.best_epoch.hdf5"
checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint,cvslog,LRS]

#%%  Run & Save
# Compile model
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
print(model.summary())
# Fit the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=128, callbacks=callbacks_list)
# Plot and save
plot_history(history)
pandas.DataFrame(history.history).to_csv("history.csv")
filepath = Full_name+'save.h5'
model.save(filepath)