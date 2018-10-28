#!/usr/bin/env python

# TODO: Rewrite this using poses

# If these imports fail, check that you ran setup_ik.py
# Instructions are in the comments of that file
# Remember to move the .so files into this folder
try:
	from ikLeft import leftIK, leftFK
	from ikRight import rightIK, rightFK
except: # If these instructions aren't enough, check the named files for more details.
	print("Did you forget to run the setup script? It's called setup_ik.py.")
	print('Run it with "python setup_ik.py build" from it\'s file location.')
	print("Maybe the .so files are in the wrong location?")
	print('They\'re called "ikLeft.so" and "ikRight.so".')
	print('Put them in the same directory as pr2_ik.py.')
	assert False, 'IK Import Failed: probably not setup'
import numpy as np
from math import pi as PI
from math import sqrt
# only needed to convert the quaternion to a matrix for the ik solver
# quaternions are in [i,j,k,r]
# set the optional parameter real_first=True to use quaternions in [r, i, j, k]
# could also be done using pybullet
from pyquaternion import Quaternion as Quat

UPPER_ARM_STEPS = 100.0 # defines how many steps to cut the upper arm range into
CURRENT_POSE_FIRST = False # if True, consider just solutions with current upper arm value first. If False, consider entire range + current all at once.

# The joint limits are taken from the pr2.urdf file. They look accurate within 0.02.
roll_joint_limits = [float('-inf'), float('inf')] # forearm and wrist roll joints have no limits. However, they will be represented using [0, 2*PI)]

LEFT_JOINT_LIMITS = [[-0.7146, 2.2854], # left shoulder pan
					 [-0.5236, 1.3963], # left shoulder lift
					 [-0.8000, 3.9000], # left upper arm roll
					 [-2.3213, 0.0000], # left elbow flex
					 roll_joint_limits, # left forearm roll (360 rotation, no limits)
					 [-2.1800, 0.0000], # left wrist flex
					 roll_joint_limits] # left wrist roll (360 rotation, no limits)

# LEFT_DEFAULT = [(limits[0] + limits[1]) / 2.0 for limits in LEFT_JOINT_LIMITS] # Not a great default because upper arm roll has a strange value
LEFT_DEFAULT = [0]*7 # a more natural position
LEFT_UPPER_ARM_LIMITS = LEFT_JOINT_LIMITS[2]
# I excluded the last step on each side to stay away from the joint limits
LEFT_UPPER_ARM_RANGE = [LEFT_UPPER_ARM_LIMITS[0] + i*(LEFT_UPPER_ARM_LIMITS[1] - LEFT_UPPER_ARM_LIMITS[0])/UPPER_ARM_STEPS for i in range(1, int(UPPER_ARM_STEPS))]

RIGHT_JOINT_LIMITS = [[-2.2854, 0.7146], # right shoulder pan
					  [-0.5236, 1.3963], # right shoulder lift
					  [-3.9000, 0.8000], # right upper arm roll
					  [-2.3213, 0.0000], # right elbow flex
					  roll_joint_limits, # right forearm roll (360 rotation, no limits)
					  [-2.1800, 0.0000], # right wrist flex
					  roll_joint_limits] # right wrist roll (360 rotation, no limits)

# RIGHT_DEFAULT = [(limits[0] + limits[1]) / 2.0 for limits in RIGHT_JOINT_LIMITS] # Not a great default because upper arm roll has a strange value
RIGHT_DEFAULT = [0]*7 # a more natural position
RIGHT_UPPER_ARM_LIMITS = RIGHT_JOINT_LIMITS[2]
# I excluded the last step on each side to stay away from the joint limits
RIGHT_UPPER_ARM_RANGE = [RIGHT_UPPER_ARM_LIMITS[0] + i*(RIGHT_UPPER_ARM_LIMITS[1] - RIGHT_UPPER_ARM_LIMITS[0])/UPPER_ARM_STEPS for i in range(1, int(UPPER_ARM_STEPS))]

# arm is 'l' or 'r' for each arm
# pos is a list of 3 values for translation [x, y, z]
# quat is a quaternion [i, j, k, r]. Pass in a list of quaternions to test multiple orientations
# torso is a float for the torso height
# current is a list of the 7 current arm joint positions
# 	the solver returns the solution which is closest to the current arm configuration
#	if none is given, it uses the middle of all the joint ranges.
# return_all is a parameter which determines how many IK solutions will be returned.
# If True, it returns a list of all legal IK solutions (possible an empty list)
# If False, it returns just the best one.
# real_first is a boolean for the format of the quaternions
# if True, quaternions use the format [r, i, j, k]
# if False, they use the format [i, j, k, r]
# dist_func is a function for ranking the IK solution candidates
# if none is provided, it uses euclidean distance with appropriate adjustments for roll joints
# if some is provided, it should be callable with dist_func(config)
# dist_func overrides current. So it uses dist_func for ranking instead of proximity to current
# filter_func is the function for a filter applied to the legal solutions
# the inputs are solutions which are within the joint limits
# if filter_func(solution) == False, that solution is excluded
# if filter_func(solution) == True, that solution is included
# This is where you would apply something like a collision checker
# the pose is in the reference frame of the baselink which should match ros_controller
# the pose is for the gripper tool frame which is exactly where the grippers meet pointing forward
def arm_ik(arm, pos, quat, torso, current=None, return_all=False, real_first=False, dist_func=None, filter_func=None):
	checkArmRanges()
	if arm == 'l':
		return left_arm_ik(pos, quat, torso, current=current, return_all=return_all, real_first=real_first, dist_func=dist_func, filter_func=filter_func)
	else:
		return right_arm_ik(pos, quat, torso, current=current, return_all=return_all, real_first=real_first, dist_func=dist_func, filter_func=filter_func)

# arm is 'l' or 'r' for each arm
# pos is a list of 8 values for the joints
# the first one is the torso and the next 7 are the arm joints
# the pose is in the reference frame of the baselink which should match ros_controller
# the pose is for the gripper tool frame which is exactly where the grippers meet pointing forward
# real_first is a boolean for the quaternion format
# if True, quaternions are in the format [r, i, j, k]
# if False, quaternions are in the format [i, j, k, r]
# torso is an optional parameter for if you want to separate the config and the torso value
# if config already contains 8 joints, torso will be ignored
# if config contains only 7 joints, torso will be added before the 7 arm joints provided
def arm_fk(arm, config, torso=None, real_first=False):
	if len(config) < 7:
		assert False, 'Not enough joint angles specified: ' + str([torso] + config)
	elif len(config) == 7:
		if torso is not None:
			config = [torso] + config
		else:
			assert False, 'Only 7 joint angles and no torso specified: ' + str(config)
	elif len(config) > 8:
		# consider making this not an error. Technically, it'll still work.
		assert False, 'Too many joint angles specified: ' + str(config)
	if arm == 'l':
		return _do_fk(config, leftFK, real_first=real_first)
	else:
		return _do_fk(config, rightFK, real_first=real_first)


def left_arm_ik(pos, quat, torso, current=None, return_all=False, real_first=False, dist_func=None, filter_func=None):
	if current is None:
		current = LEFT_DEFAULT
	if dist_func is None:
		dist_func = lambda config: arm_distance(config, current)
	limits = LEFT_JOINT_LIMITS
	upper_arm = current[2] # get position of left upper arm

	solutions = _do_ik(pos, quat, torso, upper_arm, leftIK, real_first=real_first)
	# best = get_best_solution(solutions, limits, dist_func)

	legal = get_legal_solutions(solutions, limits, filter_func=filter_func)

	if CURRENT_POSE_FIRST and return_all and len(legal):
		return legal

	best = get_closest_solution(legal, dist_func)
	bestSolutions = []

	if best is not None:
		if CURRENT_POSE_FIRST:
			return best
		else:
			if return_all:
				bestSolutions = legal
			else:
				bestSolutions.append(best)

	# IK could have failed to produce a legal solution if our upper_arm value was bad
	# So we have to iterate over the range of possible values.

	for upper_arm in LEFT_UPPER_ARM_RANGE:
		solutions = _do_ik(pos, quat, torso, upper_arm, leftIK, real_first=real_first)
		# best = get_best_solution(solutions, limits, dist_func)

		legal = get_legal_solutions(solutions, limits, filter_func=filter_func)

		if return_all:
			bestSolutions.extend(legal)
			continue
		else:
			best = get_closest_solution(legal, dist_func)
			if best is not None:
				bestSolutions.append(list(best))

	if return_all:
		return bestSolutions
	else:
		best = get_best_solution(bestSolutions, limits, dist_func, filter_func=filter_func)

		return best

def right_arm_ik(pos, quat, torso, current=None, return_all=False, real_first=False, dist_func=None, filter_func=None):
	if current is None:
		current = RIGHT_DEFAULT
	if dist_func is None:
		dist_func = lambda config: arm_distance(config, current)
	limits = RIGHT_JOINT_LIMITS
	upper_arm = current[2] # get position of right upper arm

	solutions = _do_ik(pos, quat, torso, upper_arm, rightIK, real_first=real_first)
	# best = get_best_solution(solutions, limits, dist_func)

	legal = get_legal_solutions(solutions, limits, filter_func=filter_func)

	if CURRENT_POSE_FIRST and return_all and len(legal):
		return legal

	best = get_closest_solution(legal, dist_func)
	bestSolutions = []

	if best is not None:
		if CURRENT_POSE_FIRST:
			return best
		else:
			if return_all:
				bestSolutions = legal
			else:
				bestSolutions.append(best)

	# IK could have failed to produce a legal solution if our upper_arm value was bad
	# So we have to iterate over the range of possible values.

	bestSolutions = []
	for upper_arm in RIGHT_UPPER_ARM_RANGE:
		solutions = _do_ik(pos, quat, torso, upper_arm, rightIK, real_first=real_first)
		# best = get_best_solution(solutions, limits, dist_func)

		legal = get_legal_solutions(solutions, limits, filter_func=filter_func)

		if return_all:
			bestSolutions.extend(legal)
			continue
		else:
			best = get_closest_solution(legal, dist_func)
			if best is not None:
				bestSolutions.append(list(best))

	if return_all:
		return bestSolutions
	else:
		best = get_best_solution(bestSolutions, limits, dist_func, filter_func=filter_func)

		return best

# a nicer wrapper for IK calls
# it returns a list of IK solutions with the torso value already stripped out
# pos is an [x, y, z] list for position
# quat can be either [i,j,k,r] for a single pose or [[i,j,k,r],[i,j,k,r],...] for multiple poses.
# All the parameters must be provided for the solver to work. Orientation is not optional.
def _do_ik(pos, quat, torso, upper_arm, ik_func, real_first=False):
	# the return format for ik_func is [solution1, solution2, ...] where solution1 = [torso_value, shoulder_pan_value, ...]

	solutions = []
	if isinstance(quat[0], (list, tuple)): # If quat contains multiple quaternion lists
		for rot in quat:
			rot = get_matrix_from_quat(rot, real_first=real_first)

			sol = ik_func(rot, pos, [torso, upper_arm])

			if sol is not None:
				solutions.extend(sol)

	else: # If there's just one quaternion
		rot = get_matrix_from_quat(quat, real_first=real_first)

		sol = ik_func(list(rot), list(pos), [torso, upper_arm])

		if sol is not None:
			solutions.extend(sol)

	# remove torso values
	solutions = [sol[1:] for sol in solutions]

	return solutions

# a nicer wrapper for FK calls
# returns pos, quat where pos = [x, y, z] and quat is a quaternion = [i, j, k, r]
def _do_fk(config, fk_func, real_first=False):
	pose =  fk_func(config)
	pos = pose[0]
	rot = pose[1] # 3x3 rotation matrix
	quat = Quat(matrix=np.array(rot))
	quat = quat if quat.real >= 0 else -quat # solves q and -q being same rotation
	quat = quat.unit
	quat = quat.elements.tolist()
	# switch from [r, i, j, k] to [i, j, k, r]
	if not real_first:
		quat = quat[1:] + [quat[0]]
	return pos, quat

# quat is just a list of 4 elements which represents a quaternion
# this function returns a 3x3 rotation matrix
def get_matrix_from_quat(quat, real_first=False):
	# TODO: do this in a way which doesn't require an additional library (maybe pybullet?)
	if not real_first:
		# This inversion means that the quat should be given in [i,j,k,r]. This is to match pybullet and ros.
		quat = [quat[3]] + quat[:3]
	quatObj = Quat(quat)
	return quatObj.rotation_matrix.tolist()


# returns the closest legal solution
# returns None if no legal solutions
def get_best_solution(solutions, limits, dict_func, filter_func=None):
	legal = get_legal_solutions(solutions, limits, filter_func=filter_func)
	best = get_closest_solution(legal, dict_func)
	return best

# checks all the solutions for if they violate the joint limits
# remember to strip the torso value out of the ik solution
def get_legal_solutions(solutions, limits, filter_func=None):
	# I think this actually works
	legal = [sol for sol in solutions if all([limit[0] <= s and s <= limit[1] for s, limit in zip(sol, limits)])]
	if filter_func is not None:
		legal = filter(filter_func, legal)
	return legal

	# But here's a more reasonable version
	legal = []
	for sol in solutions:
		allLegal = True
		for i in range(len(limits)):
			if sol[i] < limits[i][0] or limits[i][1] < sol[i]:
				if abs(sol[i]) < 0.00001:
					sol[i] = 0
				else:
					allLegal = False
		if allLegal and filter_func(sol):
			legal.append(sol)
	return legal

# return the closest one in joint space
# For the forearm and wrist roll joints, it adjusts the solution to be within pi of the current
# TODO: use a new distance metric which weights the different joints
def get_closest_solution(solutions, dist_func):
	minDist = float('inf')
	bestSol = None
	for sol in solutions:
		dist = dist_func(sol)
		# dist = np.linalg.norm([sol[i] - current[i] for i in range(len(sol))])
		if dist < minDist:
			minDist = dist
			bestSol = sol

	if bestSol is None:
		return None

	# All roll joints are in range [0, 2*PI]
	bestSol[4] %= 2*PI
	bestSol[6] %= 2*PI

	return bestSol

euclidean = lambda s, c: abs(s - c)
# This one is definitely correct. I think the shorter one works too though
# angular = lambda s, c: min((s - c) % (2*PI), (2*PI - ((s - c)) % (2*PI)))
angular = lambda s, c: min((s - c) % (2*PI), (c - s) % (2*PI))
distList = [euclidean]*4 + [angular, euclidean, angular]
arm_distance = lambda sol, cur: sqrt(sum(d(s, c)**2 for d, s, c in zip(distList, sol, cur)))

def checkArmRanges():
	global LEFT_UPPER_ARM_RANGE
	global RIGHT_UPPER_ARM_RANGE
	if len(LEFT_UPPER_ARM_RANGE) != UPPER_ARM_STEPS - 1:
		LEFT_UPPER_ARM_RANGE = [LEFT_UPPER_ARM_LIMITS[0] + i*(LEFT_UPPER_ARM_LIMITS[1] - LEFT_UPPER_ARM_LIMITS[0])/UPPER_ARM_STEPS for i in range(1, int(UPPER_ARM_STEPS))]
		RIGHT_UPPER_ARM_RANGE = [RIGHT_UPPER_ARM_LIMITS[0] + i*(RIGHT_UPPER_ARM_LIMITS[1] - RIGHT_UPPER_ARM_LIMITS[0])/UPPER_ARM_STEPS for i in range(1, int(UPPER_ARM_STEPS))]
		