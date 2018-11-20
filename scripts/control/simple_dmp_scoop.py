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

    self.uc.command_gripper_position(self.arm, demo_trajectory[0][0], demo_trajectory[0][1], 10)


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
      fig, ax1 = plt.subplots(1,1)
      fig2, ax2 = plt.subplots(1,1)
      fig3, ax3 = plt.subplots(1,1)
      fig4, ax4 = plt.subplots(1,1)
      fig5, ax5 = plt.subplots(1,1)
      fig6, ax6 = plt.subplots(1,1)
      fig.suptitle('X')  # Add a title so we know which it is
      fig2.suptitle('Y')  # Add a title so we know which it is
      fig3.suptitle('Z')  # Add a title so we know which it is
      fig4.suptitle('euler X')  # Add a title so we know which it is
      fig5.suptitle('euler Y')  # Add a title so we know which it is
      fig6.suptitle('euler Z')  # Add a title so we know which it is



      plan_x = [x.positions[0] for x in self.plan.plan.points]
      plan_y = [x.positions[1] for x in self.plan.plan.points]
      plan_z = [x.positions[2] for x in self.plan.plan.points]
      plan_euler_x = [x.positions[3] for x in self.plan.plan.points]
      plan_euler_y = [x.positions[4] for x in self.plan.plan.points]
      plan_euler_z = [x.positions[5] for x in self.plan.plan.points]
      times = self.plan.plan.times
      demo_x = [x[0] for x in self.demo_traj]
      demo_y = [x[1] for x in self.demo_traj]
      demo_z = [x[2] for x in self.demo_traj]
      demo_euler_x = [x[3] for x in self.demo_traj]
      demo_euler_y = [x[4] for x in self.demo_traj]
      demo_euler_z = [x[5] for x in self.demo_traj]
      times1 = [0, 5, 10, 15, 20]

      ax1.plot(times, plan_x, color='red')
      ax1.plot(times1, demo_x)

      ax2.plot(times, plan_y, color='red')
      ax2.plot(times1, demo_y)

      ax3.plot(times, plan_z, color='red')
      ax3.plot(times1, demo_z)

      ax4.plot(times, plan_euler_x, color='red')
      ax4.plot(times1, demo_euler_x)

      ax5.plot(times, plan_euler_y, color='red')
      ax5.plot(times1, demo_euler_y)

      ax6.plot(times, plan_euler_z, color='red')
      ax6.plot(times1, demo_euler_z)

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
      goal_thresh = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
      seg_length = -1
      tau = resp.tau
      dt = 1
      integrate_iter = 5

      self.plan = self.makePlanRequest(x_0, x_dot_0, t_0, goal, goal_thresh,
                            seg_length, tau, dt, integrate_iter)

  def executePlan(self):
      positions = [x.positions[0:3] for x in self.plan.plan.points]
      orientation = [quaternion_from_euler(x.positions[3],x.positions[4], x.positions[5]) for x in self.plan.plan.points]

      self.uc.command_gripper_trajectory(self.arm, positions, orientation, self.plan.plan.times)


if __name__ == '__main__':
   rospy.init_node('simple_scoop_dmp_node')
   simple_scoop = SimpleDMPScoop()
   simple_scoop.makePlan()
   #print(simple_scoop.plan)
   simple_scoop.plotPlan()
   simple_scoop.executePlan()
   #print(plan)
   
  
