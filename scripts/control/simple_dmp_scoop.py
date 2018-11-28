#!/usr/bin/env python
import roslib
roslib.load_manifest('dmp')
import rospy
import numpy as np
from dmp.srv import *
from dmp.msg import *
import matplotlib.pyplot as plt

from control_tools.ros_controller import ROS_Controller
from tf.transformations import euler_from_quaternion, quaternion_from_euler



class SimpleDMPScoop:
  def __init__(self):
    self.arm = 'r'
    self.uc =  ROS_Controller(verbose=False)
    self.plan = []

    demo_trajectory = np.load("simple_traj.npy")
    self.demo_traj = []
    for point in demo_trajectory:
      coords = point[0]
      euler = [float(x) for x in np.round(euler_from_quaternion(point[1]),2)]
      self.demo_traj.append(coords + euler)
    #self.uc.command_torso(0.0, 2, True)
    #self.uc.command_gripper_position(self.arm, demo_trajectory[0][0], demo_trajectory[0][1], 10)


  #Learn a DMP from demonstration data
  def makeLFDRequest(self, dims, traj, dt, K_gain, 
                     D_gain, num_bases):
      demotraj = DMPTraj()
          
      for i in range(len(traj)):
          pt = DMPPoint();
          pt.positions = traj[i]
          demotraj.points.append(pt)
          demotraj.times.append(dt*i)
              
      k_gains = [K_gain]*dims
      d_gains = [D_gain]*dims
          
      print "Starting LfD..."
      rospy.wait_for_service('learn_dmp_from_demo')
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
      num_bases = 100
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
     

  def executePlan(self, force_traj):
      positions = [x.positions[0:3] for x in self.plan.plan.points]
      orientation = [quaternion_from_euler(x.positions[3],x.positions[4], x.positions[5]) for x in self.plan.plan.points]
      #roughly follow position/trajectory but modify it to feel the right forces based on the phase
      for idx in range(len(positions)):
          self.uc.command_gripper_position(self.arm, positions[idx], orientation[idx], 0.3)
          phase = self.calc_phase(self.plan.plan.times[idx], self.tau)
          force_idx = int(round(phase*force_traj.shape[1]))
          current_force = force_traj[:, force_idx]
          self.feel_force(current_force)
  def feel_force(self, current_force):
     pass
  def calc_phase(self, curr_time, tau):
      return np.exp(-(self.alpha/tau)*curr_time);

if __name__ == '__main__':
   rospy.init_node('simple_scoop_dmp_node')
   simple_scoop = SimpleDMPScoop()
   simple_scoop.makePlan()
   #print(simple_scoop.plan)
   #simple_scoop.plotPlan()
   force_traj = np.zeros((6,20))
   simple_scoop.executePlan(force_traj)
   #print(plan)
   
  
