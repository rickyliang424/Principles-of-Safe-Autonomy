import rospy
from gazebo_msgs.srv import GetModelState, GetModelStateResponse
from gazebo_msgs.msg import ModelState
from ackermann_msgs.msg import AckermannDrive
import numpy as np
from std_msgs.msg import Float32MultiArray
import math
from util import euler_to_quaternion, quaternion_to_euler
import time

class vehicleController():

    def __init__(self):
        # Publisher to publish the control input to the vehicle model
        self.controlPub = rospy.Publisher("/ackermann_cmd", AckermannDrive, queue_size = 1)
        self.prev_vel = 0
        self.L = 1.75 # Wheelbase, can be get from gem_control.py
        self.log_acceleration = False
        self.lookahead_dist = 10.0

    def getModelState(self):
        # Get the current state of the vehicle
        # Input: None
        # Output: ModelState, the state of the vehicle, contain the
        #   position, orientation, linear velocity, angular velocity
        #   of the vehicle
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            serviceResponse = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            resp = serviceResponse(model_name='gem')
        except rospy.ServiceException as exc:
            rospy.loginfo("Service did not process request: "+str(exc))
            resp = GetModelStateResponse()
            resp.success = False
        return resp


    # Tasks 1: Read the documentation https://docs.ros.org/en/fuerte/api/gazebo/html/msg/ModelState.html
    #       and extract yaw, velocity, vehicle_position_x, vehicle_position_y
    # Hint: you may use the the helper function(quaternion_to_euler()) we provide to convert from quaternion to euler
    def extract_vehicle_info(self, currentPose):

        ####################### TODO: Your TASK 1 code starts Here #######################
        pos_x = currentPose.pose.position.x
        pos_y = currentPose.pose.position.y
        vx = currentPose.twist.linear.x
        vy = currentPose.twist.linear.y
        vel = math.sqrt(vx ** 2 + vy ** 2)
        _, _, yaw = quaternion_to_euler(currentPose.pose.orientation.x,
                                        currentPose.pose.orientation.y,
                                        currentPose.pose.orientation.z,
                                        currentPose.pose.orientation.w)

        ####################### TODO: Your Task 1 code ends Here #######################
        return pos_x, pos_y, vel, yaw  # note that yaw is in radian

    # Task 2: Longtitudal Controller
    # Based on all unreached waypoints, and your current vehicle state, decide your velocity
    # Task 2: Longtitudal Controller
    # Based on all unreached waypoints, and your current vehicle state, decide your velocity
    def longititudal_controller(self, curr_x, curr_y, curr_vel, curr_yaw, future_unreached_waypoints):

        ####################### TODO: Your TASK 2 code starts Here #######################

        def get_curvature(dx, dy, ddx, ddy):
            """
            Compute curvature at one point given first and second derivatives.

            :param dx: (float) First derivative along x axis
            :param dy: (float)
            :param ddx: (float) Second derivative along x axis
            :param ddy: (float)
            :return: (float)
            """
            if dx == 0 and dy == 0:
                return 0
            return (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** (3 / 2)

        def get_mapped_vel(curvature, min_curve=0.0,
                           max_curve=0.03,
                           min_vel=8.0,
                           max_vel=12.0):
            """ linear mapping """
            curvature = min(max_curve, curvature)
            curvature = max(min_curve, curvature)
            return max_vel - (max_vel - min_vel) * curvature / (max_curve - min_curve)

        n_pts = len(future_unreached_waypoints)
        ind = 0
        ori_x, ori_y = curr_x, curr_y
        curr_pt = [curr_x, curr_y]
        curvatures = []
        while True:
            tgt_pt = future_unreached_waypoints[ind]  # nearest

            next_ind = ind if ind + 1 >= n_pts else ind + 1
            next_pt = future_unreached_waypoints[next_ind]

            # compute curvature
            dx0, dy0 = tgt_pt[0] - curr_pt[0], tgt_pt[1] - curr_pt[1]
            dx1, dy1 = next_pt[0] - tgt_pt[0], next_pt[1] - tgt_pt[1]
            ddx, ddy = dx1 - dx0, dy1 - dy0
            curvature = abs(get_curvature(dx1, dy1, ddx, ddy))
            curvatures.append(curvature)

            dist = math.hypot(ori_x - tgt_pt[0], ori_y - tgt_pt[1])
            if self.lookahead_dist < dist:
                break
            if ind + 1 >= n_pts:
                break  # not exceed goal
            ind += 1
            curr_pt = tgt_pt

        avg_curvature = np.mean(curvatures)

        # if self.argv.longititudal_algo == 'linear':
        target_velocity = get_mapped_vel(avg_curvature)

        ####################### TODO: Your TASK 2 code ends Here #######################
        return target_velocity


    # Task 3: Lateral Controller (Pure Pursuit)
    def pure_pursuit_lateral_controller(self, curr_x, curr_y, curr_yaw, target_point, future_unreached_waypoints):

        ####################### TODO: Your TASK 3 code starts Here #######################
        n_pts = len(future_unreached_waypoints)
        ind = 0
        ori_x, ori_y = curr_x, curr_y
        curr_pt = [curr_x, curr_y]
        target_steerings = []
        while True:
            tgt_pt = future_unreached_waypoints[ind]  # nearest

            alpha = math.atan2(tgt_pt[1] - ori_y,
                               tgt_pt[0] - ori_x) - curr_yaw

            ld = math.hypot(tgt_pt[1] - ori_y, tgt_pt[0] - ori_x)
            target_steering = math.atan2(
                2.0 * self.L * math.sin(alpha) / ld, 1.0)
            target_steerings.append(target_steering)

            dist = math.hypot(ori_x - tgt_pt[0], ori_y - tgt_pt[1])
            if 5 < dist: #self.lookahead_dist
                break
            if ind + 1 >= n_pts:
                break  # not exceed goal
            ind += 1
            curr_pt = tgt_pt

        ####################### TODO: Your TASK 3 code starts Here #######################
        return np.mean(target_steerings)


    def execute(self, currentPose, target_point, future_unreached_waypoints):
        # Compute the control input to the vehicle according to the
        # current and reference pose of the vehicle
        # Input:
        #   currentPose: ModelState, the current state of the vehicle
        #   target_point: [target_x, target_y]
        #   future_unreached_waypoints: a list of future waypoints[[target_x, target_y]]
        # Output: None

        curr_x, curr_y, curr_vel, curr_yaw = self.extract_vehicle_info(currentPose)

        # Acceleration Profile
        if self.log_acceleration:
            acceleration = (curr_vel- self.prev_vel) * 100 # Since we are running in 100Hz
        
        print([curr_x, curr_y], ',')

        target_velocity = self.longititudal_controller(curr_x, curr_y, curr_vel, curr_yaw, future_unreached_waypoints)
        target_steering = self.pure_pursuit_lateral_controller(curr_x, curr_y, curr_yaw, target_point, future_unreached_waypoints)


        #Pack computed velocity and steering angle into Ackermann command
        newAckermannCmd = AckermannDrive()
        newAckermannCmd.speed = target_velocity
        newAckermannCmd.steering_angle = target_steering

        # Publish the computed control input to vehicle model
        self.controlPub.publish(newAckermannCmd)

    def stop(self):
        newAckermannCmd = AckermannDrive()
        newAckermannCmd.speed = 0
        self.controlPub.publish(newAckermannCmd)
