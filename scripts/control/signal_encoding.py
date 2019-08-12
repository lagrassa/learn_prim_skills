from __future__ import division
import numpy as np
from helper import plot_forces
from PIL import Image
import keras
import random
from make_plots import plot_line
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Conv2D, Flatten, Input
import ipdb
import mdp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.rcParams['font.size'] = 40
PLOT = False


def minmax_scale(x):
    return (x-np.amin(x))/(np.amax(x)-np.amin(x))
def filename_to_traj(num, name="scrape"):
    data_dims = []
    for dim in ["x", "y", "z"]:
        filename = "recordings/"+ "force_"+dim+name+str(num)+".npy" 
        dim_data= np.load(filename) 
        assert(dim_data is not None)
        dim_data += np.random.normal(0,0.05,dim_data.shape[0])
        data_dims.append(dim_data.reshape(-1,1))
    data =  np.hstack(data_dims)
    #if PLOT:
    #    plot_forces(data)
    return data

def highest_n_mag_freqs(signal, n):
    signal_fft = np.fft.fft(signal)
    freq = np.fft.fftfreq(signal.shape[-1])
    #lowest to highest
    indices_by_magnitude = sorted(list(range(len(signal_fft))), key=lambda x: signal_fft[x].real)
    highest_mag_indices = indices_by_magnitude[-n:]
    return [freq[idx] for idx in highest_mag_indices]

def print_traj_list_stats(traj_list):
    dims = ['x','y','z']
    for dim in dims:
        print("Minimum values in "+dim, np.average([min(traj[:, dims.index(dim)]) for traj in traj_list])) 
        print("Maximum values in "+dim, np.average([max(traj[:, dims.index(dim)]) for traj in traj_list])) 

    

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
            mdp.nodes.TimeFramesNode(6) +
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

def train_classifier(train_data, train_labels, test_data, test_labels, mass_data = None):
    mass_input = Input(shape=(1,), name="mass_input")
    traj_input = Input(shape=train_data[0].shape[1:], name="traj_input")
    x = keras.layers.GaussianNoise(0.8)(traj_input)
    conv_layer = Conv2D(9, (15, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D(pool_size=(3,1))(conv_layer)
    #x = Conv2D(8, (2, 2), activation='relu')(x)
    #x = keras.layers.MaxPooling2D(pool_size=(2,1))(x)
    x = Flatten()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.concatenate([x, mass_input])
    x = Dense(16, activation="relu", kernel_regularizer=keras.regularizers.l2(0.2))(x)
    x = keras.layers.Dropout(0.2)(x)
    x = Dense(8, activation="relu", kernel_regularizer=keras.regularizers.l2(0.2))(x)
    x = Dense(2,  activation='softmax')(x)
    model = Model(inputs=[traj_input, mass_input], outputs = [x])
    cortical_model = Model(inputs=[traj_input, mass_input], outputs = [conv_layer])
    model.compile(optimizer='adam',
              loss='categorical_crossentropy', metrics=['accuracy'])
    
    one_hot_train_labels =  keras.utils.to_categorical(train_labels, num_classes=2)
    one_hot_test_labels =  keras.utils.to_categorical(test_labels, num_classes=2)
    histories = []
    n_epochs= 10
    batch_size = 1
    history = model.fit(train_data, one_hot_train_labels, epochs=n_epochs, batch_size=batch_size, verbose=1, validation_data=(test_data,one_hot_test_labels))
    histories.append(history)
    if PLOT:
        n_samples = 1
        for i in range(n_samples):
            history = model.fit(train_data, one_hot_train_labels, epochs=n_epochs, batch_size=batch_size, verbose=1, validation_data=(test_data,one_hot_test_labels))
            histories.append(history)
         
    
    conv_weights = model.get_weights()[0]
    #visualize_weights(conv_weights)
    cortical_model.set_weights([conv_weights])
 
    if PLOT:
        plotting(histories)
    return model, cortical_model

def plotting(histories):
    accs = []
    val_accs = []
    for history in histories:
        accs.append(history.history['acc'])
        val_accs.append(history.history['val_acc'])
    mean_accs = np.mean(np.vstack(accs), axis=0)
    mean_val_accs = np.mean(np.vstack(val_accs), axis=0)
    stdev_accs = np.std(np.vstack(accs), axis=0)
    stdev_val_accs = np.std(np.vstack(val_accs), axis=0)
    #plt.plot(history.history['acc'], color = "red")
    #plt.plot(history.history['val_acc'], color = "blue")
    plot_line(mean_accs, stdev_accs, color="red", label="Training")
    plot_line(mean_val_accs, stdev_val_accs, color="blue", label="Test")
    #red_patch = mpatches.Patch(color='red', label='Training')
    #blue_patch = mpatches.Patch(color='blue', label='Test')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Classification Accuracy')
    plt.title("Learning curve with SFA")
    plt.ylim((0.45,1.05))
    plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0],[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.show()


    
def visualize_encoding(encoded_signals, label=""):
    import math
    #h = math.ceil(len(encoded_signals)**0.5)
    #f, axarr = plt.subplots(h,h)
    #f.suptitle(label, fontsize=20)
    i = 0
    scale = 8
    for encoded_signal in encoded_signals:
        rescaled_im = minmax_scale(encoded_signal)
        im = Image.fromarray(rescaled_im*255)
        im = im.resize((50,200))
        #col = i % h
        #row = (i - col) / h
        #surface = axarr[row, col]
        plt.imshow(im)
        plt.plot()
        plt.xlabel("Dimension")
        plt.ylabel("Time")
        plt.xticks([0, im.width/2.0, im.width],['x','y','z'])
        plt.yticks([0, im.height],[0,96])
        plt.show()
        return
        i += 1
#3,3,1,16
def visualize_weights(weights, label=""):
    import math
    h = math.ceil(weights.shape[3]**0.5)
    f, axarr = plt.subplots(h,h)
    f.suptitle(label, fontsize=20)
    i = 0
    scale = 8
    for i in range(weights.shape[3]):
        filter_im = weights[:,:,0,i]
        rescaled_im = minmax_scale(filter_im)
        im = Image.fromarray(rescaled_im*255)
        im = im.resize((10,50))
        col = int(i % h)
        row = int((i - col) / h)
        surface = axarr[row, col]
        surface.set_xticks([],[])
        surface.set_yticks([],[])
        surface.imshow(im)
    plt.show()
    
def make_dataset(outputs, traj_masses, label):
    data = np.zeros((len(outputs),)+ outputs[0].shape)
    i = 0
    for output in outputs:
        data[i, :, :] = output
        i += 1
    labels = np.array([label]*len(outputs))
    data = data.reshape(data.shape+ (1,))
    return [data, traj_masses], labels

def make_good_and_bad_dataset(num_train_good, num_train_bad, good_encoded_signals, bad_encoded_signals,  good_traj_masses, bad_traj_masses, test=False):
    if not test:
        good_data, good_labels = make_dataset(good_encoded_signals[:num_train_good], good_traj_masses[:num_train_good],1)
        bad_data, bad_labels = make_dataset(bad_encoded_signals[:num_train_bad], bad_traj_masses[:num_train_bad],  0)
    else:
        good_data, good_labels = make_dataset(good_encoded_signals[num_train_good:], good_traj_masses[num_train_good:], 1)
        bad_data, bad_labels = make_dataset(bad_encoded_signals[num_train_bad:], bad_traj_masses[num_train_bad:], 0)
    traj_data = np.vstack([good_data[0], bad_data[0]])
    mass_data = np.hstack([good_data[1], bad_data[1]])
    labels= np.hstack([good_labels, bad_labels])
    return [traj_data, mass_data], labels

"""make the encoder"""
def make_encoder(skip_sfa=False, mat_name=None):
    encoder, good_responses = test_encoding(skip_sfa=skip_sfa, test_mat=mat_name)
    return encoder, good_responses
    

def name_to_data():
    good_lookup, bad_lookup, mass = {}, {}, {}
    good_lookup["scrape"] = [0,1,2,3,4,5,6,7,8,12]
    bad_lookup["scrape"] = [9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
    mass["scrape"] = 4.5 
    good_lookup["diverse_test_"] = [1,8,9,10,14,15,16,17,18,19]
    bad_lookup ["diverse_test_"] = [2,3,4,5,6,7,20,21,22,23,24]
    mass["diverse_test_"] =2.9  
    good_lookup["small_"] = [1,2,3,4,5,6,7,11,12,13,20,21,22,23,24,25,26,27,28]
    bad_lookup ["small_"] = [8,9,14,15,16,17,18,19]
    mass["small_"] =  0.43
    good_lookup["big_"] = [2,5,6,7,8,9,16,17,18,19,20,21,22,23,26,27,30]
    bad_lookup ["big_"] =[1,3,4,11,12,13,15,24,25,28,29]
    mass["big_"] =  15.87
    return good_lookup, bad_lookup, mass
    
def unison_shuffled_copies(list_a, list_b):
    list_a = np.array(list_a)
    p = np.random.permutation(len(list_a))
    try:
        return list_a[p], list_b[p]
    except:
        ipdb.set_trace()

def get_good_bad_traj(exp_list):
    good_lookup, bad_lookup, mass = name_to_data()
    good_trajs = [] 
    bad_trajs = []
    good_traj_masses = [] 
    bad_traj_masses = [] 
    for exp in exp_list:
        exp_good_trajs = [filename_to_traj(num, name=exp) for num in good_lookup[exp]]
        exp_bad_trajs = [filename_to_traj(num, name=exp) for num in bad_lookup[exp]]
        good_trajs.extend(exp_good_trajs)
        bad_trajs.extend(exp_bad_trajs)
        mass_good_traj = np.ones(len(exp_good_trajs))*mass[exp]
        mass_bad_traj = np.ones(len(exp_bad_trajs))*mass[exp]
        good_traj_masses.extend(mass_good_traj)
        bad_traj_masses.extend(mass_bad_traj)

    #plot_forces(good_trajs)
    #:plot_forces(bad_trajs)
    
    traj_sets =  [np.array(good_trajs), np.array(bad_trajs)]
    mass_sets = [np.array(good_traj_masses), np.array(bad_traj_masses)]
    for i in range(len(traj_sets)):
        traj_sets[i], mass_sets[i] = unison_shuffled_copies(traj_sets[i], mass_sets[i])
        
    return good_trajs, good_traj_masses, bad_trajs, bad_traj_masses

def data_from_data_list(data_list, flows, split = None, skip_sfa = False, test=False):
    good_trajs, good_traj_masses, bad_trajs, bad_traj_masses = get_good_bad_traj(data_list) 
    good_encoded_signals = [apply_flows(flows, traj) for traj in good_trajs]
    #visualize_encoding(good_trajs)
    #visualize_encoding(good_encoded_signals)
    bad_encoded_signals = [apply_flows(flows, traj) for traj in bad_trajs]
    if skip_sfa:
        good_encoded_signals = good_trajs
        bad_encoded_signals = bad_trajs
    if split is None:
       split = 1
    num_train_good = int(split*len(good_trajs))
    num_train_bad = int(split*len(bad_trajs))
    data, labels = make_good_and_bad_dataset(num_train_good, num_train_bad, good_encoded_signals, bad_encoded_signals, good_traj_masses, bad_traj_masses,test=test) 
    return data, labels

def test_encoding(skip_sfa=False, test_mat = None):
    train_list = ["scrape", "diverse_test_", "small_", "big_"]
    test_list = [test_mat]
    if test_mat is not None:
        train_list.remove(test_mat)
    else:
        test_list = train_list

    #print_traj_list_stats(good_trajs)
    test_accuracies = [] 
    numtrains = 1
    split = 0.95
    for i in range(numtrains):
        good_trajs, good_traj_masses, bad_trajs, bad_traj_masses = get_good_bad_traj(train_list) 
        flows = get_flows(good_trajs)
        data, labels = data_from_data_list(train_list, flows, split=split, skip_sfa=skip_sfa) #make_good_and_bad_dataset(num_train_good, num_train_bad, good_encoded_signals, bad_encoded_signals, good_traj_masses, bad_traj_masses, test=False)
        
        test_data, test_labels = data_from_data_list(test_list, flows, split=split, skip_sfa = skip_sfa, test=test_mat is None) #make_good_and_bad_dataset(num_train_good, num_train_bad, good_encoded_signals, bad_encoded_signals, good_traj_masses, bad_traj_masses, test=True)
        model, cortical_model = train_classifier(data, labels, test_data, test_labels)
        test_accuracy = test_model(model, test_data, test_labels)
        test_accuracies.append(test_accuracy)
    print("Mean", np.mean(test_accuracies))
    print("Stdev", np.std(test_accuracies))
    print("train list", train_list)
    print("test list", test_list)
    def encoder(sample, mass):
        #put it through the signal processing model
        nerve_signal = apply_flows(flows, sample)
        #then through the cortical model
        cortical_response = apply_cortical_processing(nerve_signal, cortical_model)
        nerve_signal_data = nerve_signal.reshape((1,)+nerve_signal.shape+(1,))
        mass_data = mass*np.ones(1)
        pred = model.predict([nerve_signal_data, mass_data])
        plot_response(cortical_response)
        return cortical_response, pred
    #good_responses = [encoder(good_trajs[i], good_traj_masses[i])[0] for i in range(len(good_trajs))]
    [encoder(bad_trajs[i], bad_traj_masses[i])[0] for i in range(len(bad_trajs))]
    return encoder, good_responses
#(1, 81, 1, 9
def plot_response(res):
    arr = res[0,:,0,:]
    rescaled_im = minmax_scale(arr)
    im = Image.fromarray(rescaled_im*255)
    im = im.resize((50,200))
    #col = i % h
    #row = (i - col) / h
    #surface = axarr[row, col]
    plt.imshow(im)
    plt.plot()
    plt.show()

    


def apply_cortical_processing(nerve_signal, cortical_model):
   input_signal = nerve_signal.reshape((1,)+nerve_signal.shape+(1,))
   mass = 0.1*np.ones(1)
   return cortical_model.predict([input_signal, mass])


    
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
    import sys
    mat_name = None
    if len(sys.argv) > 1:
        mat_name = sys.argv[1]
    make_encoder(skip_sfa = False, mat_name = mat_name)
    
if __name__=="__main__":
    main()
