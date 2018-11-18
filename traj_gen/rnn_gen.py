import ipdb
import os
import tensorflow
import numpy as np
tensorflow.VERSION
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, TimeDistributed
from keras.callbacks import Callback
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


####################################################
# Plotting loss and val_loss as function of epochs #
####################################################
def plotting(history):
    plt.plot(history.history['loss'], color = "red")
    plt.plot(history.history['val_loss'], color = "blue")
    red_patch = mpatches.Patch(color='red', label='Training')
    blue_patch = mpatches.Patch(color='blue', label='Test')
    plt.legend(handles=[red_patch, blue_patch])
    plt.xlabel('Epochs')
    plt.ylabel('MSE loss')
    plt.show()


#######################
# Learning y1 from x1 #
#######################

##
# Data
##
m = 1
x = np.linspace(0,40,40)
n_data = 100
k = 0.5
train_data =np.vstack([x for i in range(n_data)])
train_labels =np.vstack([np.sin(k*x) for i in range(n_data)])
plt.plot(train_labels[0])
plt.show()

x1_train = train_data
y1_train = train_labels
inputs = x1_train.reshape(x1_train.shape+(1,))
outputs = y1_train.reshape(y1_train.shape+(1,))
inputs_test = x1_train.reshape(x1_train.shape+(1,))
outputs_test = y1_train.reshape(y1_train.shape+(1,))

##
# Model
##
model=Sequential()
dim_in = m
dim_out = m
nb_units = 10 # will also work with 2 units, but too long to train
batch_size=20

model.add(LSTM(input_shape=(None, dim_in),
                    return_sequences=True, 
                    batch_input_shape=(batch_size, None, dim_in), 
                    stateful=True, 
                    units=nb_units))
model.add(TimeDistributed(Dense(activation='linear', units=dim_out)))
model.compile(loss = 'mse', optimizer = 'adam')

##
# Training
##
# 2 seconds for each epoch
np.random.seed(1337)
history = model.fit(inputs, outputs, epochs = 2000, batch_size = batch_size,
                    validation_data=(inputs_test, outputs_test))
plotting(history)
plt.plot(model.predict(inputs[:20])[0])
plt.show()
