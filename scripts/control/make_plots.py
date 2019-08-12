import matplotlib.pyplot as plt
import os
import ast
import csv
import os
import numpy as np
import pdb
plt.rcParams['font.size'] = 18

#Import data
#get list of means and stdevs

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot_line(mean, stdev, color="red", label="missing label", plot_area = None,xaxis=None, n=1) :
    y = mean
    #smooth  
    y_above = [mean[i]+stdev[i]/n for i  in range(mean.shape[0])]
    y_below = [mean[i]-stdev[i]/n for i  in range(mean.shape[0])]
    display_now = False
    if plot_area is None:
        display_now = True
        plot_area = plt
    #plot mean
    if xaxis is None:
        coords = list(range(len(mean)))
    else:
        coords = xaxis
    plot_area.plot(coords, y, label=label, color=color)
    plot_area.fill_between(coords, y_below, y_above, color=color, alpha = 0.3)


def get_stdev_and_mean(exp_list, prefix, root_dir = "No root directory", cutoff=None, lengths_array = None):

    if lengths_array is None:
        lengths_list = []
        for exp in exp_list:
            lengths = get_line_out_file(prefix+exp, root_dir = root_dir)
            lengths_list.append(lengths)
        try:
            shortest_length = min([len(l) for l in lengths_list])
        except: 
            pdb.set_trace()
        if cutoff is not None:
            shortest_length = min(cutoff, shortest_length)
        short_length_list = [l[:shortest_length]for l in lengths_list]
        lengths_array = np.vstack(short_length_list)
    stdevs = np.std(lengths_array, axis=0)
    means = np.mean(lengths_array, axis=0)
    return means, stdevs

"""This list of keys will appear in the legend, the list is experiment names
This should plot the average lengths, and then rewards"""
def plot_graph(exp_dict, 
              prefix="no prefix", 
              title="No title",
              xlab = "No x label", 
              root_dir = "No root directory",
              plot_area = None,
              cutoff=None,
              lengths_array_index=None,
              ylab = "No y label"):
    #First plot average lengths
    colors = ["red", "blue","green", "purple", "gray", "yellow" ]
    color_i = 0
    for exp_name in exp_dict.keys():
        if lengths_array_index is None:
            means, stdevs = get_stdev_and_mean(exp_dict[exp_name], prefix, root_dir = root_dir, cutoff=cutoff)
        else:
            means, stdevs = get_stdev_and_mean(exp_dict[exp_name], prefix, root_dir = root_dir, cutoff=cutoff, lengths_array=exp_dict[exp_name][lengths_array_index])
        plot_line(means, 2*stdevs, color = colors[color_i], label=exp_name, plot_area = plot_area, n = len(exp_dict[exp_name]))
        color_i +=1 
    plot_area.set_xlabel(xlab)
    plot_area.set_ylabel(ylab)

"""Plots 2 plots on top of each other: average reward
and average length of episode"""
def plot_learning_curve(exp_dict, title = "No title", root_dir="No root directory", cutoff=None):
    #plot average length on top        
    f, axarr = plt.subplots(2,sharex=True)
    plt.title(title)
    plot_graph(exp_dict, prefix = "rewards",root_dir = root_dir,  xlab = "Number of episodes", ylab = "Average episode reward", plot_area=axarr[0], cutoff=cutoff)
    plot_graph(exp_dict, prefix = "diversity",root_dir = root_dir,  xlab = "Number of episodes", ylab = "Average episode length", plot_area=axarr[1], cutoff=cutoff)
    #Then rewards
    plt.legend()
    plt.show()

def plot_policy_test(exp_dict,root_dir = "", title = "No title", cutoff=None):
    #plot average length on top        
    f, axarr = plt.subplots(2,sharex=True)
    plt.title(title)
    plot_graph(exp_dict, prefix = "average_length",root_dir = root_dir,  xlab = "Number of steps in episodes", ylab = "Hierarchical entropy", plot_area=axarr[0], cutoff=cutoff, lengths_array_index = 0)
    plot_graph(exp_dict, prefix = None,root_dir = root_dir,  xlab = "Number of steps in episode", ylab = "Ratio beads in cup", plot_area=axarr[1], cutoff=cutoff, lengths_array_index = 1)
    #Then rewards
    plt.legend()
    plt.show()

def get_line_out_file(exp, root_dir = "No root directory"):
    if exp[-3:] == 'npy':
        float_list = np.load(root_dir+exp)
    else:
        with open(root_dir+exp, 'rb') as csvfile:
            reader = csv.reader(csvfile)
            string_list =  reader.next()
            try:
                float_list =  [float(elt) for elt in string_list if elt != ""]
            except:
                pdb.set_trace()
    smoothed = moving_average(float_list, n = 3)
    return smoothed

def get_exps_from_root(root, root_dir="stats/"):
    #finds all experiments with root in the name and a number
    files = os.listdir(root_dir)
    filenames = []
    for f in files:
        if root+"_"in f and "reward" in f and ".pyc" not in f:
            filenames.append(f[len("rewards"):])
    return filenames

def generate_exp_dictionary_one_vars(root):
    exp_dict = {}
    deltas = {"0dot1":0.1, "0dot01":0.01, "0dot05":0.05}
    for delta_name in deltas:
        try:
            exp_dict["delta="+str(deltas[delta_name])] = get_exps_from_root(root+"_del_"+delta_name)
        except:
            print("Did not find file name")
            pdb.set_trace()
    return exp_dict
        
def plot_torque_exps():
    mean_force = np.array([20.010048707706765, 20.033980977696054, 20.044033163759998,20.04328196195048, 20.07714936284933, 20.119015514165458])
    std_force = np.array([1.183963437353486, 0.017352829218864173, 0.02569722121784631, 0.029303600402577304, 0.02779617487145231, 0.02976682680187362])
    std_torque = np.array([0.001027841957337963, 0.0006618554335287074, 0.0007878713611964948, 0.0011958797597496223, 0.0012688585153662107, 0.0012225045480852425])
    mean_torque = np.array([1.1525293608164768, 1.1548105236681214, 1.1580053209347017,1.1629171611970301, 1.1733667603331372,  1.183963437353486])
    num_beads =  [0,1,2,4,8,12]
    f, axarr = plt.subplots(2,sharex=True)
    plt.title("Mean force and torques over number of beads")
    plot_line(mean_force, std_force, color="red", label="Force", plot_area = axarr[0],xaxis=num_beads, n=1)
    plot_line(mean_torque, std_torque, color="blue", label="Torque", plot_area = axarr[1],xaxis=num_beads, n=1)
    axarr[0].set_ylabel("Force magnitude")
    axarr[1].set_ylabel("Torque magnitude")
    plt.show()

def plot_torque_and_force(root="recordings/", exp_name="scoop"):
    _, axarr = plt.subplots(8,sharex=True)
    num_plots = 20
    successes = [0,1,2,3,4,5,6,7,8,12]
    dims = ["x", "y", "z"]
    for measurement_type in ["force", "torque"]:
        for j in range(1,num_plots):
            torque_mags = None
            force_mags = None
            #TODO color based on success
            if j in successes:
                color = 'g'
            else:
                color = 'r'
            for dim in dims:
                name = measurement_type+"_"+dim+exp_name
                data = load_time_series(name, num_plots, root=root)
                if force_mags is None:
                    torque_mags = np.zeros(data[0,:].shape)
                    force_mags = np.zeros(data[0,:].shape)
                i = dims.index(dim) 
                if measurement_type == "torque":
                    i += 3
                axarr[i].plot(data[j, :], color=color)
                if measurement_type == "torque":
                    torque_mags += data[j,:]**2
                else:
                    force_mags += data[j,:]**2

                if j == num_plots-1:
                    if measurement_type == "torque":
                        if "x" in name:
                            name = name.replace('x', 'roll')
                        elif "y" in name:
                            name = name.replace('y', 'pitch')
                        elif "z" in name:
                            name = name.replace('z', 'yaw')
                      
                    name = name.replace(exp_name, '')
                    name = name.replace('_', ' ')
  
                    axarr[i].set_ylabel(name)
            torque_mags = torque_mags**0.5
            plt.xlabel("Time steps")
            force_mags = force_mags**0.5
            if measurement_type == "force":
                axarr[6].plot(force_mags, color=color)
                axarr[6].set_ylabel("force \n magnitude")
            else:
                axarr[7].plot(torque_mags, color=color)
                axarr[7].set_ylabel("torque \n magnitude")
    plt.show()
    
                
            
""" returns a 10x array of all of the lines that fit that format
Each row is one of the lines, calibrated to the shortest"""            
def load_time_series(name, num_plots, root = "recordings/"):
    all_data = np.zeros((num_plots,100))
    for i in range(1,num_plots):
        plot_name  = name+str(i)+".npy"
        new_line = np.load(root+plot_name)
        all_data[i,:] = new_line
    return all_data
        
    
        
def main():
    plot_torque_and_force(root="../erase_controller_pkg/scripts/control/recordings/", exp_name="scrape")
  
if __name__ == "__main__":            
    tip = {}
    tip["5 init solve for yaw"] = get_exps_from_root("known_yaw", root_dir="data/") 
    #tip["400 to init unknown yaw"] = get_exps_from_root("400_initial_long_no_tip_contexts", root_dir="data/") 
    #plot_learning_curve(tip, title = "Penalizing bowl tipping results", root_dir="data/", cutoff=180)
    main()
   

    #plot_learning_curve(generate_exp_dictionary_one_vars("pdg_one_var_stepsize"), title = "Step size one variable", root_dir="stats/", cutoff=None)
    #plot_learning_curve(generate_exp_dictionary_one_vars("pdg_one_var_desheight"), title = "desired height one variable", root_dir="stats/", cutoff=None)
    #plot_learning_curve(generate_exp_dictionary_one_vars("pdg_one_var"), title = "Offset one variable", root_dir="stats/", cutoff=None)
    #plot_learning_curve(generate_exp_dictionary_one_vars("pdg_one_var"), title = "Offset one variable", root_dir="stats/", cutoff=None)
    #plot_learning_curve(generate_exp_dictionary_one_vars("pdg_one_var_dt"), title = "dt one variable", root_dir="stats/", cutoff=None)
    #plot_learning_curve(generate_exp_dictionary_one_vars("pdg_one_var_force"), title = "force one variable", root_dir="stats/", cutoff=None)



