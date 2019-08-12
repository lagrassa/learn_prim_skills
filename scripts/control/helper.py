import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

#forces is 3 columns and n_traj points (rows)
def plot_forces(forces):
    colors = ['r', 'g', 'b']
    labels = ['x', 'y', 'z']
    for i in range(3):
        plt.plot(forces[:, i], color = colors[i], label=labels[i])
    plt.xlabel("timestep")
    plt.ylabel("magnitude")
    plt.legend()
    plt.show()

def smooth_traj(traj, n=3):
    smoothed = np.zeros(traj.shape)
    for i in range(smoothed.shape[1]):
        averaged = moving_avg(traj[:,i])
        padding = smoothed[:,i].shape[0]-averaged.shape[0]
        averaged = np.hstack([averaged, np.zeros((padding))])
        smoothed[:,i] = averaged
    return smoothed

def moving_avg(traj, n = 5):
    ret = np.cumsum(traj, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def subsample_traj(traj, n=5):
    return traj[::n,:]
