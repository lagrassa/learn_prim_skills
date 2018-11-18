#from signal_encoding import  encode_signal
from __future__ import division
from functools import reduce
import operator
import ipdb
import matplotlib.pyplot as plt
import signal_encoding
import numpy as np
PLOT = False
"""
('Minimum values in x', -1.8303492891940318)
('Maximum values in x', 15.884808416405301)
('Minimum values in y', -10.074215694184076)
('Maximum values in y', 0.4013318639148463)
('Minimum values in z', -15.65198031694467)
('Maximum values in z', -2.2643545356492645)

"""

lower = [-1.83,-10.07,-15.6]
upper = [15.88,0.4,-2.26]

def rbf(x, width, center):
   return np.exp(-width * (x-center)**2) 
   #coeff = 1.0/(sigma*(2*np.pi)**0.5) 
   #return coeff*np.exp((-0.5*((x-center)/sigma))**2)

"""not going to make these params meaningful currently
sum of N 3D gaussians weighted by weight to get a n_traj x 3 trajectory
"""
def gen_parameterized_forces(weights, n_traj, N = 3):
    #could be a sum of basis functions, where the params are the weights
    #make a multivariate RBG 3 dimensions 
    centers_list = np.linspace(0,n_traj, N)
    #rbfs = [lambda x: np.exp(-0.05 * (x - center)**2) for center in centers ] 
    result = np.zeros((n_traj, 3))
    #add each of the RBFs with n_traj points
    center_span =0.08#0.02
    for dim in range(3): #for each dimension
        for i in range(len(centers_list)): #for each center
            grid_result = np.zeros((n_traj, 3)) 
            for traj in range(n_traj): #for each point
                grid_result[traj, dim] = rbf(traj, center_span, centers_list[i])
            try:
                result += weights[dim, i]*grid_result
            except:
                print("Some sort of shape error", "weights are ", weights.shape)
                ipdb.set_trace()
    rescaled_result = rescale_to_constraints(result)
    return result #rescaled_result


#forces is 3 columns and n_traj points (rows)
def plot_forces(forces):
    colors = ['r', 'g', 'b']
    labels = ['x', 'y', 'z']
    for i in range(3):
        plt.plot(forces[:, i], color = colors[i], label=labels[i])
    plt.legend()
    plt.show()

def distance(signal, compare_signals):
    num_points = reduce(operator.mul, signal.shape, 1)
    dists = []
    for compare_signal in compare_signals:
        dist =  np.linalg.norm(signal-compare_signal)/num_points
        dists.append(dist)
    return np.average(dists)

def gen_weights(N=3):
    #should go roughly from -5 to 11
    dim_range = np.vstack([lower, upper])
    force_param_per_dim = []
    for dim in range(3):
        a,b = dim_range[0, dim], dim_range[1,dim]
        force_params_dim = (b - a) * np.random.random_sample((N)) + a
        force_param_per_dim.append(force_params_dim)
    force_params = np.vstack([force_param_per_dim])
    #usually best for them to be mostly sparse actually, so erase each with probability p
    p = 0.8
    p_matrix = np.random.random((force_params.shape))
    mask = p_matrix > p
    force_params_sparse = np.where(mask, force_params, np.zeros(mask.shape))
    return force_params_sparse

def rescale_to_constraints(force_params_sparse):
    #then normalize to be within expected ranges
    force_params_rescaled = np.zeros(force_params_sparse.shape)
    for dim in range(3):
        #normalize to lower and upper
        arr = force_params_sparse[:, dim]
        force_params_rescaled[:, dim] = np.interp(arr, (arr.min(), arr.max()), (lower[dim], upper[dim]))
    return force_params_rescaled

def find_best_encoding(N=20):
    n_traj = 100  
    num_iters=6000
    encoder, good_responses = signal_encoding.make_encoder()
    force_to_dist = {}
    force_list = [] 
    class_list = [] 
    dist_list = [] 
    for i in range(num_iters):
        weights = gen_weights(N=N)
        forces = gen_parameterized_forces(weights, n_traj)
        #print("Min", np.min(forces.flatten()))
        #print("Max", np.max(forces.flatten()))
        cortical_response, classification_vector = encoder(forces) #we want the second term to be 1
        dist = distance(cortical_response, good_responses)
        prediction = classification_vector[0][1]
        force_list.append(forces)
        class_list.append(prediction)
        force_to_dist[i] = dist
        dist_list.append(dist)
    #best_i = min(force_to_dist.keys(), key=lambda x: force_to_dist[x])
    best_i = np.argmax(class_list)
    lowest_dist = force_to_dist[best_i]
    print("lowest dist", lowest_dist)
    print("highest class", class_list[best_i])
    if PLOT:
        plt.scatter(dist_list, class_list)
        plt.xlabel("Distance")
        plt.ylabel("Probability successful | theta")
        plt.show() 
    
    return force_list[best_i]
    
N=30 
#plot_forces(gen_parameterized_forces(gen_weights(N=N), 100, N=N))
plot_forces(find_best_encoding(N=N))
    
    
    
