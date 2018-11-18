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


m = 3
batch_size=20
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

def get_data(simple=False):
    if simple:
        x = np.linspace(0,100,100)
        n_data = 40
        k = 0.1
        train_data = np.zeros((n_data, 100,m))
        for dim in range(3):      
            train_data[:,:,dim]= np.vstack([x for _ in range(n_data)])

        train_labels = np.zeros(train_data.shape)#np.vstack([np.sin(k*x) for i in range(n_data)])
        for dim in range(3):
            new_train_labels =np.vstack([np.sin(k*(1+dim)*x) for i in range(n_data)])
            train_labels[:,:,dim] = new_train_labels
            plt.plot(train_labels[0,:,dim], label=str(dim))
        plt.show()
        x1_train = train_data
        y1_train = train_labels
        return x1_train, y1_train


def make_model():
    model=Sequential()
    dim_in = m
    dim_out = m
    nb_units = 10 # will also work with 2 units, but too long to train

    model.add(LSTM(input_shape=(None, dim_in),
                        return_sequences=True, 
                        batch_input_shape=(batch_size, None, dim_in), 
                        stateful=True, 
                        units=nb_units))
    model.add(TimeDistributed(Dense(activation='linear', units=dim_out)))
    model.compile(loss = 'mse', optimizer = 'rmsprop')
    return model

def train_model(model, x1_train, y1_train):
    np.random.seed(1337)
    inputs = x1_train.reshape(x1_train.shape)
    outputs = y1_train.reshape(y1_train.shape)
    inputs_test = x1_train.reshape(x1_train.shape)
    outputs_test = y1_train.reshape(y1_train.shape)
    history = model.fit(inputs, outputs, epochs = 5, batch_size = batch_size,
                        validation_data=(inputs_test, outputs_test))
    return model

#plotting(history)
def predict(model, inputs):
    inputs = inputs.reshape(inputs.shape)
    prediction = model.predict(inputs[:batch_size])[0]
    plt.plot(prediction)
    plt.show()

def get_traj(input_vec):
    model.predict(inputs[:batch_size])

def main():
    x_train, y_train = get_data(simple=True)
    model = make_model()
    model = train_model(model, x_train, y_train)
    predict(model, x_train)
    

if __name__ == "__main__":
    main()
