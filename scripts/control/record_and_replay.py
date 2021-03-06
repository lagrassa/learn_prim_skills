#!/usr/bin/env python
import sys
import pdb
import rospy 
import numpy as np
from control_tools.ros_controller import ROS_Controller as UberController
from geometry_msgs.msg import WrenchStamped

rospy.init_node("shakey_record") 

uc = UberController()

def record_angles(total_time = 15):
    print("recording!!!!")
    print("recording!!!!")
    print("recording!!!!")
    print("recording!!!!")
    print("recording!!!!")
    print("recording!!!!")
    print("recording!!!!")
    step_size = 0.2
    torso = uc.command_torso(0.3, 4, True)
    times = []
    time = total_time
    angles = []
    while time >= 0:
        angle = uc.adjust_config(uc.get_joint_positions(uc._get_joint_names('r')), uc.get_arm_positions('r'))
        angles.append(angle)
        times.append(total_time-time)
        rospy.sleep(step_size)
        time -= step_size
    np.save("recordings/angles.npy", angles)
    np.save("recordings/torso.npy", torso)
    np.save("recordings/times.npy", times)

def record_torques(fraction=1.0, total_time=15):
    print("Recording")
    print("Recording")
    print("Recording")
    print("Recording")
    print("Recording")
    xs = []
    ys = []
    zs = []
    forces = []    
    rs = []
    ps = []
    yaws = []
    torques = []
    step_size=0.2
    for i in range(int(total_time/step_size)):
        ws = rospy.wait_for_message("/ft/r_gripper_motor",WrenchStamped) 
        xs.append(ws.wrench.force.x)
        ys.append(ws.wrench.force.y)
        zs.append(ws.wrench.force.z)
        rs.append(ws.wrench.torque.x)
        ps.append(ws.wrench.torque.y)
        yaws.append(ws.wrench.torque.z)
        forces.append((ws.wrench.force.x**2+ws.wrench.force.y**2+ws.wrench.force.z**2)**0.5)
        torques.append((ws.wrench.torque.x**2+ws.wrench.torque.y**2+ws.wrench.torque.z**2)**0.5)
        rospy.sleep(step_size)
    #print("average x", np.mean(xs), "std x", np.std(xs))
    #print("average y", np.mean(ys), "std y", np.std(ys))
    #print("average z", np.mean(zs), "std z", np.std(zs))
    print("average force", np.mean(forces), "std z", np.std(forces))
    #print("average r", np.mean(rs), "std r", np.std(rs))
    #print("average p", np.mean(ps), "std p", np.std(ps))
    #print("average yaw", np.mean(yaws), "std yaw", np.std(yaws))
    print("average torque", np.mean(torques), "std z", np.std(torques))
    np.save("recordings/force_x"+str(fraction)+".npy", xs)
    np.save("recordings/force_y"+str(fraction)+".npy", ys)
    np.save("recordings/force_z"+str(fraction)+".npy", zs)
    np.save("recordings/torque_x"+str(fraction)+".npy", rs)
    np.save("recordings/torque_y"+str(fraction)+".npy", ps)
    np.save("recordings/torque_z"+str(fraction)+".npy", yaws)
    print("All recordings saved")
    print("All recordings saved")
    print("All recordings saved")
    print("All recordings saved")
    print("All recordings saved")
    


def replay():
    angles = np.load("recordings/angles.npy")
    times = np.load("recordings/times.npy")
    torso = np.load("recordings/torso.npy")
    angles = uc.adjust_trajectory(angles, uc.get_arm_positions('r'))
    uc.command_arm_trajectory('r', angles, times, True)

if __name__ == "__main__":
    name = sys.argv[1]
    record_torques(name, total_time=20)
#
        
