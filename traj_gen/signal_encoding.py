from __future__ import division
import numpy as np
from PIL import Image
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Flatten
import ipdb
import mdp
import matplotlib.pyplot as plt


def filename_to_traj(num, name="scrape"):
    data_dims = []
    for dim in ["x", "y", "z"]:
        filename = "recordings/"+ "force_"+dim+name+str(num)+".npy" 
        dim_data= np.load(filename) 
        dim_data += np.random.normal(0,0.1,dim_data.shape[0])
        data_dims.append(dim_data.reshape(-1,1))
    return np.hstack(data_dims)

def highest_n_mag_freqs(signal, n):
    signal_fft = np.fft.fft(signal)
    freq = np.fft.fftfreq(signal.shape[-1])
    #lowest to highest
    indices_by_magnitude = sorted(list(range(len(signal_fft))), key=lambda x: signal_fft[x].real)
    highest_mag_indices = indices_by_magnitude[-n:]
    return [freq[idx] for idx in highest_mag_indices]
    

def test_highest_n_mag_freqs():
    num_samples = 5000
    x = np.arange(num_samples)
    a = 0.1
    sine_wave = np.sin(a*x)
    f = a/(2*np.pi)
    #4/2pi had better be the top frequency
    res = highest_n_mag_freqs(sine_wave, 4)
    assert(abs(res[-1] - f) < 0.01)
    print("Test passed")

def most_common_freqs_list(signal_list, n=5, n_out = 10):
    top_freqs_all = None
    for signal in signal_list:
        top_freqs = highest_n_mag_freqs(signal, n)
        if top_freqs is None:
            top_freqs_all = top_freqs
        else:
            top_freqs_all = np.hstack([top_freqs_all, top_freqs])
    unique, counts = np.unique(top_freqs_all, return_counts=True)
    freq_to_count = dict(zip(unique, counts))
    freqs_least_common_to_most = sorted(unique, key=lambda x: freq_to_count[x])
    return freqs_least_common_to_most[-n_out:]
#spectrogram of various bins. apply a filter 
def cochlear_encoding():
    freq_range = [0.02, 0.1]

def plot_spectrograms(signal_list, label = "No title"):
    #make a figure and an axarr
    f, axarr = plt.subplots(len(signal_list))
    f.suptitle(label, fontsize=20)
    for i in range(len(signal_list)):
        surface = axarr[i]
        spectrum, freqs, t, im = surface.specgram(signal_list[i], NFFT=8, Fs = int(20), noverlap=0, pad_to=None, mode="magnitude", color='b')
 

def plot_slow(signal_list, flow, label = "No title"):
    #make a figure and an axarr
    f, axarr = plt.subplots(len(signal_list))
    f.suptitle(label, fontsize=20)
    for i in range(len(signal_list)):
        surface = axarr[i]
        surface.plot(flow(signal_list[i].reshape(-1,1)), color='b')
        surface.plot(signal_list[i], color='r')

"list of flows for each dim"
def sfa(signal_list):
    #put into format where columns are variables and rows are observations
    ndims = signal_list[0].shape[1]
    flows = []
    for i in range(ndims):
        flow = (mdp.nodes.EtaComputerNode() +
            mdp.nodes.TimeFramesNode(5) +
            mdp.nodes.PolynomialExpansionNode(2) +
            mdp.nodes.SFANode(output_dim=1, include_last_sample=True) +
            mdp.nodes.EtaComputerNode() )
        for signal in signal_list:
            signal_to_train_on = signal[:,i]
            flow.train(signal_to_train_on.reshape(-1,1))
        flows.append(flow)
    return flows
    
        
def apply_flows(flows, signal):
    outputs = []
    for i in range(len(flows)):
        flow = flows[i]
        output = flow(signal[:,i].reshape(-1,1))
        outputs.append(output)
    return np.hstack(outputs)
    
     
def get_flows(example_signal):
    x = np.linspace(0,50, 3000)
    x += np.random.normal(1,0.9,3000)
    #simple_traj = np.hstack([np.sin(x), np.sin(5*x), np.sin(0.5*x)])
    simple_traj= (x-15)**2*(x-2)**2*(x-35)**2*(x-50)**2
    flows= sfa(example_signal)
    return flows

def train_classifier(data, labels):
    model = Sequential()
    model.add(Conv2D(32, (2, 2), activation='relu', input_shape=data.shape[1:]))
    cortical_model = Sequential()
    cortical_model.add(Conv2D(32, (2, 2), activation='relu', input_shape=data.shape[1:]))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(2,  activation='softmax'))
    model.compile(optimizer='adam',
              loss='categorical_crossentropy', metrics=['accuracy'])
    cortical_model.compile(optimizer='adam',
              loss='categorical_crossentropy', metrics=['accuracy'])
    conv_weights = model.get_weights()[0]
    cortical_model.set_weights([conv_weights])
    one_hot_labels =  keras.utils.to_categorical(labels, num_classes=2)
    model.fit(data, one_hot_labels, epochs=10, batch_size=5, verbose=1)
    return model, cortical_model


    
def visualize_encoding(encoded_signals, label=""):
    import math
    h = math.ceil(len(encoded_signals)**0.5)
    f, axarr = plt.subplots(h,h)
    f.suptitle(label, fontsize=20)
    i = 0
    scale = 8
    for encoded_signal in encoded_signals:
        im = Image.fromarray(encoded_signal*255)
        im = im.resize((40,40))
        col = i % h
        row = (i - col) / h
        surface = axarr[row, col]
        surface.imshow(im)
        i += 1
    
def make_dataset(outputs, label):
    data = np.zeros((len(outputs),)+ outputs[0].shape)
    i = 0
    for output in outputs:
        data[i, :, :] = output
        i += 1
    labels = np.array([label]*len(outputs))
    data = data.reshape(data.shape+ (1,))
    return data, labels

def make_good_and_bad_dataset(num_train_good, num_train_bad, good_encoded_signals, bad_encoded_signals, flows, test=False):
    if not test:
        good_data, good_labels = make_dataset(good_encoded_signals[:num_train_good], 1)
        bad_data, bad_labels = make_dataset(bad_encoded_signals[:num_train_bad], 0)
    else:
        good_data, good_labels = make_dataset(good_encoded_signals[num_train_good:], 1)
        bad_data, bad_labels = make_dataset(bad_encoded_signals[num_train_bad:], 0)
    data = np.vstack([good_data, bad_data])
    labels= np.hstack([good_labels, bad_labels])
    return data, labels

"""make the encoder"""
def make_encoder():
    encoder, good_response = test_encoding()
    return encoder, good_response
    
def test_encoding():
    successes = [0,1,2,3,4,5,6,7,8,12]
    fails = [9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
    good_trajs = [filename_to_traj(num) for num in successes]
    bad_trajs = [filename_to_traj(num) for num in fails]
    flows = get_flows(good_trajs)
    good_encoded_signals = [apply_flows(flows, traj) for traj in good_trajs]
    bad_encoded_signals = [apply_flows(flows, traj) for traj in bad_trajs]
    num_train_good = 7
    num_train_bad = 10
    data, labels = make_good_and_bad_dataset(num_train_good, num_train_bad, good_encoded_signals, bad_encoded_signals, flows, test=False)
    model, cortical_model = train_classifier(data, labels)
    test_data, test_labels = make_good_and_bad_dataset(num_train_good, num_train_bad, good_encoded_signals, bad_encoded_signals, flows, test=False)
    print ("accuracy", test_model(model, test_data, test_labels))
    def encoder(sample):
        #put it through the signal processing model
        nerve_signal = apply_flows(flows, sample)
        #then through the cortical model
        cortical_response = apply_cortical_processing(nerve_signal, cortical_model)
        return cortical_response
    good_response = encoder(good_trajs[0])
    return encoder, good_response


def apply_cortical_processing(nerve_signal, cortical_model):
   input_signal = nerve_signal.reshape((1,)+nerve_signal.shape+(1,))
   return cortical_model.predict(input_signal)
    
def test_model(model, test_data, test_labels):
    predictions = model.predict(test_data)
    num_correct = 0
    for  i in range(predictions.shape[0]):
        pred = predictions[i]
        best = np.argmax(pred)
        actual = test_labels[i]
        if best == actual:
            num_correct += 1 
    return num_correct/predictions.shape[0]
        
    #model_pred = model.predict(test_data)
    #num_incorrect = sum(abs(model_pred - test_labels))
    #total_num = test_data.shape[0]
    #return (total_num-num_incorrect)/(total_num)

def main():
    #This aims to find an encoding that forms a spectrogram that is visibly different between good force trajectories and bad ones
    make_encoder()
    


main()
