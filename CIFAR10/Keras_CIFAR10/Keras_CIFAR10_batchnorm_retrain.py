from __future__ import print_function
import pandas
import keras
from matplotlib import pyplot as plt
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger

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
    
#%% Data Preparation
# Load data
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

#%% Params Model 
epochs = 50
lr = 0.01
momentum = 0.9
factor_red = 0.2
patience_red = 25
sgd = SGD(lr=lr, momentum=momentum, nesterov=True)
Model_name = '2Stack_3CONV_3FC_'
Model_optim = 'sgd_lr%smom%sepoch%s_' % (lr,momentum,epochs) 
Model_reduclr = 'fc%spat%s_' % (factor_red,patience_red)
Full_name = Model_name+Model_optim+Model_reduclr
filepath = 'test_save.h5'
model = keras.models.load_model(filepath)

lr = 0.007

#%% Callbacks
# Checkpoint
checkpoint_filepath = Full_name+"weights2.best_epoch.hdf5"
checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# Reduce LR
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=factor_red, patience=patience_red, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=1e-5)
# CVSlog
csvlog_filepath = Full_name+"csvlog2.csv"
cvslog = CSVLogger(csvlog_filepath, separator=',', append=False)
callbacks_list = [checkpoint,cvslog,reduce_lr]

#%% Run & Save
# Compile Model
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())
# Fit the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=128, callbacks=callbacks_list)
# Plot and save
plot_history(history)
pandas.DataFrame(history.history).to_csv(Full_name+"history2.csv")
filepath = Full_name+'save2.h5'
model.save(filepath)