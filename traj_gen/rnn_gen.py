import ipdb
import os
import tensorflow
import numpy as np
tensorflow.VERSION
from keras.models import load_model
from helper import smooth_traj, subsample_traj, plot_forces
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, TimeDistributed, GRU, ConvLSTM2D
from keras.callbacks import Callback
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.rcParams['font.size'] = 18
from signal_encoding import get_good_bad_traj, make_good_and_bad_dataset

PLOT=False
m = 1
ndim = 3
batch_size=1
look_back=5
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

def gen_timeline(n_data, n_traj):
    x = np.linspace(0,n_traj,n_traj)
    train_data = np.zeros((n_data, n_traj,m))
    for dim in range(3):      
        train_data[:,:,dim]= np.vstack([x for _ in range(n_data)])
    return train_data
#time, look_back, 1
def create_dataset(dataset, look_back=1):
        num_points = len(dataset)*(dataset[0].shape[0]-look_back-1)
        trainX = np.zeros((num_points, 3,look_back))
        trainY = np.zeros((num_points,3,1))
        data_num = 0
        for i in range(len(dataset)):
                traj = dataset[i]
                for j in range(len(traj)-look_back-1):
                    #blocks of look_back
                    curr_block = traj[j:j+look_back, :].T
                    next_block = traj[j+look_back+1]
                    trainX[data_num, :, :] = curr_block
                    trainY[data_num, :, :] = next_block.reshape(next_block.shape+(1,))
                    data_num += 1
                    
            
        return trainX,  trainY


def get_data(simple=False):
    if simple:
        n_traj = 20
        train_data = gen_timeline(40, n_traj)
        x = np.linspace(0,n_traj,n_traj)
        train_labels = np.zeros(train_data.shape)#np.vstack([np.sin(k*x) for i in range(n_data)])
        n_data = 40
        k = 0.05
        for dim in range(3):
            new_train_labels =np.vstack([np.sin(k*(1+dim)*x) for i in range(n_data)])
            train_labels[:,:,dim] = new_train_labels
        #plot_forces(train_labels[0, :,:])
        #x1_train = train_data
        trainX, trainY =create_dataset(train_data, look_back=look_back)
        return trainX, trainY
        #y1_train = train_labels
        #return y1_train, y1_train
    else:
        good_trajs, bad_trajs = get_good_bad_traj()
        good_trajs = [smooth_traj(traj, n=5) for traj in good_trajs]
        bad_trajs = [smooth_traj(traj, n=5) for traj in bad_trajs]
        
        good_trajs = [subsample_traj(traj, n=5) for traj in good_trajs]
        bad_trajs = [subsample_traj(traj, n=5) for traj in bad_trajs]
        n_traj = good_trajs[0].shape[0]
        print("traj shape", np.array(good_trajs).shape)
        if PLOT:
            plot_forces(good_trajs[2])
        num_train_good = int(0.75*len(good_trajs))
        num_train_bad = int(0*len(bad_trajs))
        #data, _ = make_good_and_bad_dataset(num_train_good, num_train_bad, good_trajs, bad_trajs, test=False)
        #data = np.zeros((num_train_good+num_train_bad, n_traj, 3)) #HACK 
        #for i in range(num_train_good+num_train_bad):
        #    data[i, :, :] = good_trajs[0]
        #timeline = gen_timeline(data.shape[0], n_traj)
        #data = data[:,:,:,0]
        #data shifted back by repeating the first one lag+1 times
        trainX, trainY =create_dataset(good_trajs[:num_train_good], look_back=look_back)
        
        return trainX, trainY
        train_data = np.zeros(data.shape)
        first_row = data[:,0,:]
        lag =1
        for i in range(lag):
            train_data[:,i,:] =data[:,i,:]#first_row
        #and the rest
        train_data[lag:,:,:] = data[:lag,:,:]
        return train_data, data
        #test_data, test_labels = make_good_and_bad_dataset(num_train_good, num_train_bad, good_encoded_signals, bad_encoded_signals, flows, test=False

def make_model():
    model=Sequential()
    dim_in = m
    dim_out = m
    nb_units = 30# will also work with 2 units, but too long to train
    model.add(LSTM(input_shape=(batch_size,3,look_back),
                        return_sequences=True, 
                        batch_input_shape=(batch_size,3, look_back), 
                        stateful=True, 
                        units=nb_units))
    #model.add(ConvLSTM2D(11, (5,2), stateful=True, return_sequences=False, batch_input_shape=(batch_size, 34, look_back, dim_in, 1), input_shape=(None, 34, look_back, dim_in, 1)) )
    model.add(TimeDistributed(Dense(activation='linear', units=dim_out)))
    model.compile(loss = 'mse', optimizer = 'rmsprop')
    return model

def train_model(model, x1_train, y1_train):
    np.random.seed(1337)
    inputs = x1_train.reshape(x1_train.shape)
    outputs = y1_train.reshape(y1_train.shape)
    inputs_test = x1_train.reshape(x1_train.shape)
    outputs_test = y1_train.reshape(y1_train.shape)
    history = model.fit(inputs, outputs, epochs = 2, batch_size = batch_size,shuffle=False, 
                        validation_data=(inputs_test, outputs_test))
    return model

#plotting(history)
def predict(model, inputs):
    inputs = inputs.reshape(inputs.shape)
    full_prediction = []
    curr_set= np.zeros((1,3,5))
    full_prediction = None
    ipdb.set_trace()
    for i in range(13):
	curr_set[:,:,:-1]= curr_set[:,:,1:]
	curr_set[:,:,-1]= model.predict(curr_set)[0].T
        ipdb.set_trace()
        if full_prediction is None:
            full_prediction = curr_pt
        else:
            full_prediction = np.hstack([full_prediction, curr_pt])
    plot_forces(full_prediction)

def get_traj(input_vec):
    model.predict(inputs, batch_size=batch_size)


def main():
    x_train, y_train = get_data(simple=False)
    model = make_model()
    model = train_model(model, x_train, y_train)
    predict(model, x_train)
    

if __name__ == "__main__":
    main()
