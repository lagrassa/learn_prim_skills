#from signal_encoding import  encode_signal
from __future__ import division
from functools import reduce
import operator
import ipdb
import matplotlib.pyplot as plt
import signal_encoding
import numpy as np

def rbf(x, width, center):
   return np.exp(-width * (x-center)**2) 

"""not going to make these params meaningful currently
sum of N 3D gaussians weighted by weight to get a n_traj x 3 trajectory
"""
def gen_parameterized_forces(weights, n_traj, N = 3):
    length = 4
    #could be a sum of basis functions, where the params are the weights
    #make a multivariate RBG 3 dimensions 
    centers_list  = []
    center_one_dim = np.linspace(0,n_traj, N)
    for x in center_one_dim:
        for y in center_one_dim:
            for z in center_one_dim:
                centers_list.append([x,y,z])
    centers = np.vstack(centers_list)
    #rbfs = [lambda x: np.exp(-0.05 * (x - center)**2) for center in centers ] 
    input_space = np.linspace(0,95, 100)
    result = np.zeros((n_traj, 3))
    for i in range(len(centers)):
        grid_result = np.zeros((n_traj, 3)) 
        for traj in range(n_traj):
            grid_result[traj] = rbf(traj, 0.02, centers[i])
        result += weights[i]*grid_result
    return result


#forces is 3 columns and n_traj points (rows)
def plot_forces(forces):
    colors = ['r', 'g', 'b']
    labels = ['x', 'y', 'z']
    for i in range(3):
        plt.plot(forces[:, i], color = colors[i], label=labels[i])
    plt.show()

def distance(signal, compare_signals):
    num_points = reduce(operator.mul, signal.shape, 1)
    dists = []
    for compare_signal in compare_signals:
        dist =  np.linalg.norm(signal-compare_signal)/num_points
        dists.append(dist)
    return np.average(dists)

def gen_weights():
    N = 20    
    #should go roughly from -5 to 11
    a = -4
    b = 5
    force_params = (b - a) * np.random.random_sample((N**3)) + a
    return force_params

def find_best_encoding():
    n_traj = 100  
    num_iters=2000
    encoder, good_responses = signal_encoding.make_encoder()
    force_to_dist = {}
    force_list = [] 
    for i in range(num_iters):
        weights = gen_weights()
        forces = gen_parameterized_forces(weights, n_traj)
        #print("Min", np.min(forces.flatten()))
        #print("Max", np.max(forces.flatten()))
        response = encoder(forces)
        dist = distance(response, good_responses)
        force_list.append(forces)
        force_to_dist[i] = dist
    best_i = min(force_to_dist.keys(), key=lambda x: force_to_dist[x])
    lowest_dist = force_to_dist[best_i]
    print("lowest dist", lowest_dist)
    return force_list[best_i]
    
        
#plot_forces(gen_parameterized_forces(force_params, 100, N=N))
find_best_encoding()
    
    
    
