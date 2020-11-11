import os
import rospy
import argparse
import numpy as np
import math
from math import pi
import random
import gym

from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from gazebo_msgs.srv import SpawnModel, DeleteModel
from hector_nav_msgs.srv import GetRobotTrajectory
from visualization_msgs.msg import Marker

from rosgraph_msgs.msg import Clock

epi = 1e-6
diagonal_dis = math.sqrt(2) * (3.6 + 3.8)
goal_model_dir = "/home/tortes/turtle_ws/src/turtlebot3_simulations/turtlebot3_gazebo/models/Target/model.sdf"



class Env():
    def __init__(self, args):
        self.position = Pose()
        self.goal_position = Pose()
        self.goal_position.position.x = 0.
        self.goal_position.position.y = 0.
        self.marker = Marker()
        self.marker.header.frame_id = "odom"
        self.marker.type = 2
        self.marker.color.b = 1
        self.marker.color.g = 1
        self.marker.color.a = 1
        self.marker.scale.x = .1
        self.marker.scale.y = .1
        self.marker.scale.z = .1
        self.marker.pose = self.goal_position

        self.linear_velocity = args.linear_velocity
        self.angular_velocity = args.angular_velocity
        self.realsense_max_point = args.realsense_max_point
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.sub_lidar = rospy.Subscriber('scan', LaserScan, self.getLidar)

        self.time = 0
        self.sub_time = rospy.Subscriber('clock', Clock, self.getClock)

        self.goal_pub = rospy.Publisher('goal', Marker, queue_size=10)

        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.goal = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.del_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        self.clear_trajectory = rospy.ServiceProxy('/trajectory_clear', GetRobotTrajectory)

        self.action_space()
        self.observation_space()
        
        self.blocked = False
        self.past_distance = 0.
        if args.is_training:
            self.threshold_arrive = 0.2
        else:
            self.threshold_arrive = 0.4

    def getGoalDistace(self):
        goal_distance = math.hypot(self.goal_position.position.x - self.position.x, self.goal_position.position.y - self.position.y)
        self.past_distance = goal_distance

        return goal_distance

    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        q_x, q_y, q_z, q_w = orientation.x, orientation.y, orientation.z, orientation.w
        yaw = round(math.degrees(math.atan2(2 * (q_x * q_y + q_w * q_z), 1 - 2 * (q_y * q_y + q_z * q_z))))

        if yaw >= 0:
             yaw = yaw
        else:
             yaw = yaw + 360

        rel_dis_x = round(self.goal_position.position.x - self.position.x, 1)
        rel_dis_y = round(self.goal_position.position.y - self.position.y, 1)

        # Calculate the angle between robot and target
        if rel_dis_x > 0 and rel_dis_y > 0:
            theta = math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x > 0 and rel_dis_y < 0:
            theta = 2 * math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x < 0 and rel_dis_y < 0:
            theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x < 0 and rel_dis_y > 0:
            theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x == 0 and rel_dis_y > 0:
            theta = 1 / 2 * math.pi
        elif rel_dis_x == 0 and rel_dis_y < 0:
            theta = 3 / 2 * math.pi
        elif rel_dis_y == 0 and rel_dis_x > 0:
            theta = 0
        else:
            theta = math.pi
        rel_theta = round(math.degrees(theta), 2)

        diff_angle = abs(rel_theta - yaw)

        if diff_angle <= 180:
            diff_angle = round(diff_angle, 2)
        else:
            diff_angle = round(360 - diff_angle, 2)

        self.rel_theta = rel_theta
        self.yaw = yaw
        self.diff_angle = diff_angle

    def getClock(self, clock):
        self.time = clock.clock.secs
        return clock.clock.secs

    def getLidar(self, raw_data):
        scan_data = []
        for data in raw_data.ranges:
            if data == float('Inf'):
                scan_data.append(3.5)
            elif np.isnan(data):
                scan_data.append(0)
            else:
                scan_data.append(data)
        self.scan_data = scan_data

    def getState(self):
        scan_range = self.scan_data
        yaw = self.yaw
        rel_theta = self.rel_theta
        diff_angle = self.diff_angle
        min_range = 0.2
        done = False
        arrive = False

        current_distance = math.hypot(self.goal_position.position.x - self.position.x, self.goal_position.position.y - self.position.y)
        done = True if (min_range > min(scan_range) > 0) else False
        arrive = True  if current_distance <= self.threshold_arrive else False

        return scan_range, current_distance, yaw, rel_theta, diff_angle, done, arrive

    def setReward(self, done, arrive, ang_vel):
        current_distance = math.hypot(self.goal_position.position.x - self.position.x, self.goal_position.position.y - self.position.y)
        distance_rate = (self.past_distance - current_distance)

        reward = 2*distance_rate
        self.past_distance = current_distance
        
        # reward = reward - 0.1 * abs(ang_vel)
        reward = reward - 0.1

        if done:
            reward = -5.

        if arrive:
            reward = 150.
            arrive = False

        return reward

    def step(self, action):
        self.goal_pub.publish(self.marker)

        linear_vel = self.action_space_discrete[action][0]
        ang_vel = self.action_space_discrete[action][1]

        # Execute
        vel_cmd = Twist()
        if self.blocked:
            vel_cmd.linear.x = 0
            vel_cmd.angular.z = 0.5
        else:
            vel_cmd.linear.x = linear_vel / 4
            vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        scan_range, rel_dis, yaw, rel_theta, diff_angle, done, arrive = self.getState()
        self.blocked = done

        # Normailization
        scan_range = [i / 3.5 for i in scan_range]
        state_info = [rel_dis / diagonal_dis, yaw / 360, rel_theta / 360, diff_angle / 180]
        state = scan_range + state_info
        state_dict = {
            "scan_range": np.asarray(scan_range),
            "state_info": np.asarray(state_info)
        }

        reward = self.setReward(done, arrive, ang_vel)
        time_limit = True if self.time > 600 else False        
        # if arrive:
        #     print("Arrival")
        # if time_limit:
        #     print("TLE")
            

        return state, reward, arrive or time_limit, {}

    def reset(self):
        # Reset model
        rospy.wait_for_service('/gazebo/delete_model')
        try:
            self.del_model('target')
        except:
            pass
        
        # Reset simulation
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        # Reset Trajectory
        rospy.wait_for_service('trajectory_clear')
        try:
            self.clear_trajectory()
        except (rospy.ServiceException) as e:
            print("trajectory_clear service call failed")

        # Build the targetz
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        try:
            goal_urdf = open(goal_model_dir, "r").read()
            target = SpawnModel
            target.model_name = 'target'  # the same with sdf name
            target.model_xml = goal_urdf
            self.goal_position.position.x, self.goal_position.position.y = self.goal_on_law()
            self.goal(target.model_name, target.model_xml, 'namespace', self.goal_position, 'world')
            self.marker.pose = self.goal_position
        except (rospy.ServiceException) as e:
            print("/gazebo/failed to build the target")
        
        rospy.wait_for_service('/gazebo/unpause_physics')

        self.goal_distance = self.getGoalDistace()
        scan_range, rel_dis, yaw, rel_theta, diff_angle, done, arrive = self.getState()
        scan_range = [i / 3.5 for i in scan_range]
        state_info = [rel_dis / diagonal_dis, yaw / 360, rel_theta / 360, diff_angle / 180]
        state = scan_range + state_info
        state_dict = {
            "scan_range": np.asarray(scan_range),
            "state_info": np.asarray(state_info)
        }

        return state

    def goal_on_law(self):
        x_ = 0
        y_ = 0

        while True:
            x_ = random.uniform(-1.8, 1.8)
            y_ = random.uniform(-1.8, 1.8)
            
            dist1 = math.hypot(x_+0.6, y_+0.6)
            dist2 = math.hypot(x_+0.6, y_-0.6)
            dist3 = math.hypot(x_-0.6, y_-0.6)
            dist4 = math.hypot(x_-0.6, y_+0.6)

            if (dist1 > 0.2) and (dist2 > 0.2) and (dist3 > 0.2) and (dist4 > 0.2):
                break
        
        return x_, y_
                
    def action_space(self):
        action_space_discrete_ = []
        linear_sample = np.arange(0,self.linear_velocity+epi,
                                    self.linear_velocity/4.,
                                    dtype=np.float32).tolist()
        angular_sample = np.arange(-self.angular_velocity, self.angular_velocity+epi,
                                    self.angular_velocity*2/10,
                                    dtype=np.float32).tolist()
        
        for linear_speed in linear_sample:
            for angular_speed in angular_sample:
                action_space_discrete_.append([linear_speed, angular_speed])
        
        self.action_space_discrete = action_space_discrete_
        self.action_dim = len(action_space_discrete_)
        return gym.spaces.Discrete(len(action_space_discrete_))
    
    def observation_space(self):
        self.state_dim = 4 + self.realsense_max_point
        return gym.spaces.Box(low=0, high=1, shape=(self.state_dim,), dtype=np.float32)

    def seed(self, seed):
        return seed
    
    def render(self):
        return 0
    
    def close(self):
        return 0