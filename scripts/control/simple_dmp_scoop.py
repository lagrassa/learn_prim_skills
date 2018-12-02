#!/usr/bin/env python
import roslib
from pr2_ik import arm_ik
roslib.load_manifest('dmp')
import rospy
from geometry_msgs.msg import Point, WrenchStamped
import numpy as np
from dmp.srv import *
from dmp.msg import *
import sys
import matplotlib.pyplot as plt
sys.path.append("../../traj_gen/")
from gen_good_signal import find_best_encoding

from control_tools.ros_controller import ROS_Controller
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import ipdb



class SimpleDMPScoop:
    def __init__(self):
      self.arm = 'r'
      print("Waiting for DMP service...")
      rospy.wait_for_service('learn_dmp_from_demo')
      print(" DMP service up!@")
      self.uc =  ROS_Controller(verbose=False)
      self.plan = []
      self.demo_trajectory = np.load("simple_traj.npy")
      self.demo_traj = []
      self.demo_poses = []
      self.demo_orns = []
      for point in self.demo_trajectory:
	coords = point[0]
        self.demo_poses.append(point[0])
        self.demo_orns.append(point[1])
	euler = [float(x) for x in np.round(euler_from_quaternion(point[1]),2)]
	self.demo_traj.append(coords + euler)
      self.uc.command_torso(0.3, 5, True)
      self.uc.command_gripper_position(self.arm, self.demo_trajectory[0][0], self.demo_trajectory[0][1], 10)


    #Learn a DMP from demonstration data
    def makeLFDRequest(self, dims, traj, dt, K_gain, 
		       D_gain, num_bases):
	demotraj = DMPTraj()
	    
	for i in range(len(traj)):
	    pt = DMPPoint();
	    pt.positions = traj[i]
	    demotraj.points.append(pt)
	    demotraj.times.append(dt*i)
		
	k_gains = [600, 600, 600, 850, 850, 850]
        d_gains = [2 * np.sqrt(item) for item in k_gains]
        d_gains[3] *= 1.001
        d_gains[4] *= 1.001
        d_gains[5] *= 1.001
	    
	print "waiting for service"
	rospy.wait_for_service('learn_dmp_from_demo')
	print "service up!"

	try:
	    lfd = rospy.ServiceProxy('learn_dmp_from_demo', LearnDMPFromDemo)
	    resp = lfd(demotraj, k_gains, d_gains, num_bases)
	except rospy.ServiceException, e:
	    print "Service call failed: %s"%e
	print "LfD done"    
		
	return resp;


    #Set a DMP as active for planning
    def makeSetActiveRequest(self, dmp_list):
	try:
	    sad = rospy.ServiceProxy('set_active_dmp', SetActiveDMP)
	    sad(dmp_list)
	except rospy.ServiceException, e:
	    print "Service call failed: %s"%e


    #Generate a plan from a DMP
    def makePlanRequest(self, x_0, x_dot_0, t_0, goal, goal_thresh, 
			seg_length, tau, dt, integrate_iter):
	print "Starting DMP planning..."
	rospy.wait_for_service('get_dmp_plan')
	try:
	    gdp = rospy.ServiceProxy('get_dmp_plan', GetDMPPlan)
	    resp = gdp(x_0, x_dot_0, t_0, goal, goal_thresh, 
		       seg_length, tau, dt, integrate_iter)
	except rospy.ServiceException, e:
	    print "Service call failed: %s"%e
	print "DMP planning done"   
		
	return resp;


    def plotPlan(self):
	fig, axarr = plt.subplots(6)
	fig.suptitle('DMP plan v. demo trajectory')
	dims = ["x", "y", "z", "roll", "pitch", "yaw"]
	dim_poses = [[j.positions[i] for j in self.plan.plan.points] for i in range(len(dims))]
	demo_poses = [[j[i] for j in self.demo_traj] for i in range(len(dims))]
	times = self.plan.plan.times
	times1 = [0, 5, 10, 15, 20]
	for i in range(len(axarr)):
	    ax = axarr[i]
	    ax.plot(times,dim_poses[i], color="red") 
	    ax.plot(times1,demo_poses[i], color="blue") 
	    ax.set_ylabel(dims[i])
	plt.xlabel("Time")
	plt.show()

	return

    def writePlanToFile(self, write_file="simple_scoop_plan.txt"):
        f = open(write_file, "w+")
        f.write("plan start:")
        for point in self.plan.plan.points:
            f.write(str(point.positions) + "\n")
        f.write("plan complete \n\n")
        f.close()

    def makePlan(self):
        #scoop trajectory
        dims = 6
        dt = 5.0
        K = 100
        D = 2 * np.sqrt(K)
        num_bases = 500
        traj = self.demo_traj
        resp = self.makeLFDRequest(dims, traj, dt, K, D, num_bases)
        #Set it as the active DMP
        self.makeSetActiveRequest(resp.dmp_list)
        #x_0 = self.uc.get_arm_positions('r')
        gripper_pos, gripper_quat = self.uc.return_cartesian_pose(self.arm, 'base_link')
        euler = np.round(euler_from_quaternion(gripper_quat),2)
        #x_0 = gripper_pos + [x for x in euler]
        x_0 = self.demo_traj[0]
        x_dot_0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        t_0 = 0
        goal = self.demo_traj[-1]
        goal_thresh = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        seg_length = 25#-1
        self.tau = resp.tau
        self.alpha = -np.log(0.01) #for 99% convergence at t = tau
        dt = 1
        integrate_iter = 1#5
        self.plan = self.makePlanRequest(x_0, x_dot_0, t_0, goal, goal_thresh,
                            seg_length, self.tau, dt, integrate_iter)
     
    def collect_forces(self, look_back, sim=True):
        if sim:
            sample_traj = np.array([[ 11.79592668,  -5.7967972 ,  -4.85384078],
       [ 11.82233935,  -5.82096504,  -4.79712684],
       [ 11.78509921,  -5.83467317,  -4.82370593],
       [ 11.81205824,  -5.79180465,  -4.76800181],
       [ 11.85105279,  -5.87950617,  -4.95814986],
       [ 11.77549207,  -5.84337369,  -4.8196947 ],
       [ 11.7104716 ,  -5.82018938,  -4.84383969],
       [ 11.48606115,  -5.95501947,  -4.77804068],
       [ 11.83056461,  -5.91592864,  -4.82269888],
       [ 11.68511725,  -5.90105298,  -4.98439957],
       [ 11.84945954,  -5.95835304,  -4.82190914],
       [  7.17501448,  -0.18549415,  -2.3133373 ],
       [ 13.55169726, -11.31809861,  -3.78676747],
       [ 13.2013189 ,  -8.64674159,  -6.19304597],
       [ 11.0580735 ,  -3.83173013,  -5.95654665]])
            curr_forces = sample_traj[:look_back, :].T
            curr_forces = curr_forces.reshape((1,)+curr_forces.shape)
            return curr_forces

        force_list = []
        for i in range(look_back):
            force_data = rospy.wait_for_message("/ft/r_gripper_motor",WrenchStamped).wrench.force
            force_i = [force_data.x, force_data.y, force_data.z]
            force_list.append(force_i)
        return np.vstack(force_list)

    def executePlan(self, force_traj):
        positions = [x.positions[0:3] for x in self.plan.plan.points]
        orientation = [quaternion_from_euler(x.positions[3],x.positions[4], x.positions[5]) for x in self.plan.plan.points]
        look_back = 5
        current_forces = self.collect_forces(look_back=look_back)
        force_traj = find_best_encoding(curr_forces = current_forces)
        #roughly follow position/trajectory but modify it to feel the right forces based on the phase
        simple=False
        if simple:
            positions = self.demo_poses
            orientation = self.demo_orns
            times = self.uc.generate_times(positions, total_time=20.0)
            self.uc.command_gripper_trajectory('r', positions, orientation, times)
        else:
	    for idx in range(len(positions)):
                command_pose(self.uc, (positions[idx], orientation[idx]), self.arm, timeout=1.5, unwind=True)
	        #self.uc.command_gripper_position(self.arm, positions[idx], orientation[idx], 1.5)
	        phase = self.calc_phase(self.plan.plan.times[idx], self.tau)
	        force_idx = int(round(phase*force_traj.shape[1]))
	        current_force = force_traj[force_idx, :]
	        #self.feel_force(current_force)

    def feel_force(self, current_force):
        kp = 0.01
        gripper_pos, gripper_quat = self.uc.return_cartesian_pose(self.arm, 'base_link')
        force_to_feel = current_force
        for i in range(2):
            ws = rospy.wait_for_message("/ft/r_gripper_motor",WrenchStamped)
            felt_force = [ws.wrench.force.x, ws.wrench.force.y, ws.wrench.force.z]
            error =  np.array(force_to_feel-felt_force)
            if np.linalg.norm(error) < 6:
                break;
            correction = kp*error
            self.shift(correction, forced_gripper_pos=gripper_pos)

    def shift(self,correction, yaw=None, forced_gripper_pos=None):
        shift_time = 0.2
        print(correction, "correction") 
        gripper_pos, gripper_quat = self.uc.return_cartesian_pose(self.arm, 'base_link')
        if forced_gripper_pos is not None:
            ds = correction
            for i in range(3):
                if ds[i] == 0:
                    gripper_pos[i] = forced_gripper_pos[i]


        if yaw is not None:
            gripper_euler = list(euler_from_quaternion(gripper_quat))
            gripper_euler[2] = yaw
            gripper_quat = list(quaternion_from_euler(gripper_euler[0], gripper_euler[1], gripper_euler[2]))
        new_pos = gripper_pos + correction
        command_pose(self.uc, (new_pos, gripper_quat), self.arm, timeout=shift_time)

    def calc_phase(self, curr_time, tau):
        return np.exp(-(self.alpha/tau)*curr_time);

def command_pose(uc, pose, arm, timeout=4, unwind=True):
    uc.command_gripper_position(arm,pose[0], pose[1], timeout=timeout, blocking=False)
    return 
    gripper_pos, gripper_quat = uc.return_cartesian_pose(arm, 'base_link')
    positions = [gripper_pos, pose[0]]
    orientations = [gripper_quat, pose[1]]
    times = uc.generate_times(positions)    
    #uc.command_gripper_trajectory(arm, positions, orientations, times)
    #angles =  arm_ik('r', pose[0], pose[1], uc.get_torso_position(), current=uc.get_arm_positions('r'))
    #uc.command_arm('r', angles,timeout,  blocking=True)

if __name__ == '__main__':
   rospy.init_node('simple_scoop_dmp_node')
   simple_scoop = SimpleDMPScoop()
   simple_scoop.makePlan()
   #print(simple_scoop.plan)
   #simple_scoop.plotPlan()
   force_traj = np.zeros((6,20))
   simple_scoop.executePlan(force_traj)
   #print(plan)
