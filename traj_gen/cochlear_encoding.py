import numpy as np
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
    
     

def main():
    #This aims to find an encoding that forms a spectrogram that is visibly different between good force trajectories and bad ones
    successes = [0,1,2,3,4,5,6,7,8,12]
    fails = [9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
    good_traj = [filename_to_traj(num) for num in successes]
    bad_traj = [filename_to_traj(num) for num in fails]
    x = np.linspace(0,50, 3000)
    x += np.random.normal(1,0.9,3000)
    #simple_traj = np.hstack([np.sin(x), np.sin(5*x), np.sin(0.5*x)])
    simple_traj= (x-15)**2*(x-2)**2*(x-35)**2*(x-50)**2
    flows= sfa(good_traj)
    output = apply_flows(flows, good_traj[0])

    #print(flow(good_traj[0]))
    #plot_slow(good_traj, flow, label = "Good trajectories")
    #plot_slow(bad_traj, flow, label = "Bad trajectories")
    #plt.show()
    
    
    #take the top 10 from every set, then pick the mode of that
    #first let's just start on the z axis
    #print out the most salient frequencies 
    

main()
