#!/home/tortes/anaconda3/envs/py3/bin/python
import os
import rospy
import numpy as np
import math
from math import pi
import random

from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan, PointCloud2, Imu, NavSatFix
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from gazebo_msgs.srv import SpawnModel, DeleteModel

from rosgraph_msgs.msg import Clock
import sensor_msgs.point_cloud2 as pc2 # pcl lib
# from velodyne_msgs.msg import VelodyneScan, VelodynePacket

EARTH_RADIUS = 6378137
REALSENSE_MAX_POINT = 16000
zero_point = (0,0,0)
diagonal_dis = math.sqrt(2) * 100
goal_model_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], '..', '..', 'turtlebot3_simulations',
                              'turtlebot3_gazebo', 'models', 'Target_col', 'model.sdf')

# Velodyne Disabled

class Env():
    def __init__(self, is_training):
        self.position = Pose()
        self.goal_position = Pose()
        self.goal_position.position.x = 0.
        self.goal_position.position.y = 0.
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.sub_odom = rospy.Subscriber('odometry/filtered', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.goal = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.del_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)

        self.time = 0
        self.roll = 0.
        self.pitch = 0.
        self.nav_yaw = 0.
        # self.realsense_point = []
        # self.scan_data = []
        # self.nav_position = []
        self.sub_time = rospy.Subscriber('clock', Clock, self.getClock)
        self.sub_imu = rospy.Subscriber('imu/data', Imu, self.getQuaternion)
        self.sub_realsense = rospy.Subscriber('realsense/downsample', PointCloud2, self.getRealsense)
        self.sub_lidar = rospy.Subscriber('scan', LaserScan, self.getLidar)
        self.sub_navsat = rospy.Subscriber('navsat/fix', NavSatFix, self.getNavSat)

        self.past_distance = 0.
        self.nav_goal_distance = 0.
        self.nav_rel_theta = 0.
        self.nav_diff_angle = 0.

        if is_training:
            self.threshold_arrive = 0.2
        else:
            self.threshold_arrive = 0.4

    def getGoalDistace(self):
        goal_distance = math.hypot(self.goal_position.position.x - self.position.x, self.goal_position.position.y - self.position.y)
        self.past_distance = goal_distance

        return goal_distance

    def getNavGoalDistance(self):
        nav_goal_distance = math.hypot(self.goal_position.position.x - self.nav_position[0], self.goal_position.position.y - self.nav_position[1])
        self.nav_past_distance = nav_goal_distance

        return nav_goal_distance

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
        # return clock.clock.secs

    def getQuaternion(self, imu_data):
        # roll, pitch, yaw
        q_data = imu_data.orientation
        eular_data = self.getEular(q_data)
        self.orientation = q_data
        self.roll = eular_data[0]
        self.pitch = eular_data[1]
        self.nav_yaw = eular_data[2]

    def getRealsense(self, realsense_data):
        rs_generator = pc2.read_points(realsense_data, skip_nans=True, field_names=("x","y","z"))
        realsense_point_ = list(rs_generator)
        rs_point_length = len(realsense_point_)
        realsense_point_.extend([zero_point for _ in range(REALSENSE_MAX_POINT-rs_point_length)])
        self.realsense_point = realsense_point_ 

    def getLidar(self, scan_raw_data):
        scan_data_ = []
        scan_length = len(scan_raw_data.ranges)
        for i in range(scan_length):
            if scan_raw_data.ranges[i] == float('Inf'):
                scan_data_.append(30.)
            elif np.isnan(scan_raw_data.ranges[i]):
                scan_data_.append(0)
            else:
                scan_data_.append(scan_raw_data.ranges[i])
        self.scan_data = scan_data_

    def getNavSat(self, navsat_data):
        # reference Longi:45 Lati:45
        ref_longi = 45.0
        ref_lati = 45.0

        longitude = navsat_data.longitude
        latitude = navsat_data.latitude

        delta_longi = (longitude-ref_longi) * pi / 180
        delta_lati = (latitude-ref_lati) * pi / 180

        para_longi = 0.5 * (1-math.cos(delta_longi))
        para_lati = math.cos(latitude*pi/180) * math.cos(latitude*pi/180)
        if delta_longi >= 0:
            para_symbol = 1
        else:
            para_symbol = -1

        longitude_aff = para_symbol * EARTH_RADIUS * math.acos(1-2*para_lati*para_longi)
        latitude_aff = EARTH_RADIUS * delta_lati

        self.nav_position = [longitude_aff, latitude_aff]

    def getGoalAngle(self):
        rel_dis_x = round(self.goal_position.position.x - self.nav_position[0], 1)
        rel_dis_y = round(self.goal_position.position.y - self.nav_position[1], 1)

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

        diff_angle = abs(rel_theta - self.nav_yaw)

        if diff_angle <= 180:
            diff_angle = round(diff_angle, 2)
        else:
            diff_angle = round(360 - diff_angle, 2)

        self.nav_rel_theta = rel_theta
        self.nav_diff_angle = diff_angle

    def getState(self):
        self.getGoalAngle()

        realsense_data = self.realsense_point
        roll = self.roll
        pitch = self.pitch
        yaw = self.nav_yaw
        rel_theta = self.nav_rel_theta
        diff_angle = self.nav_diff_angle

        min_range = 0.3 # Collision range
        done = False
        arrive = False

        # Add physical factors to depend done
        done = self.is_done()

        # Use only dist to depend arrive
        current_distance = math.hypot(self.goal_position.position.x - self.nav_position[0], self.goal_position.position.y - self.nav_position[1])
        if current_distance <= self.threshold_arrive:
            arrive = True

        # Temp version
        extend_data = []
        for point in realsense_data:
            extend_data.extend([point[0],point[1],point[2]])

        # Check data
        assert len(extend_data) == 48000

        return extend_data, current_distance, roll, pitch, yaw, rel_theta, diff_angle, done, arrive

    def setReward(self, done, arrive):
        current_distance = math.hypot(self.goal_position.position.x - self.nav_position[0], self.goal_position.position.y - self.nav_position[1])
        distance_rate = (self.past_distance - current_distance)

        reward = 500.*distance_rate
        self.past_distance = current_distance

        # Time reward
        # reward = reward - .5 * self.time

        # Imu reward
        if abs(self.roll) > 22.5:
            # print("Alert! Roll angle is %.2f" % self.roll)
            reward = reward - 1.
        if abs(self.pitch) > 22.5:
            # print("Alert! Pitch angle is %.2f" % self.pitch)
            reward = reward - 1.

        if done:
            reward = -100.
            self.pub_cmd_vel.publish(Twist())

        if arrive:
            reward = 120.
            self.pub_cmd_vel.publish(Twist())
            arrive = False

        return reward

    def step(self, action, past_action):
        linear_vel = action[0]
        ang_vel = action[1]

        vel_cmd = Twist()
        vel_cmd.linear.x = linear_vel / 4
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)
        
        # Update sensor data
        # self.getSensor()

        # Update state observation
        realsense_data, rel_dis, roll, pitch, yaw, rel_theta, diff_angle, done, arrive = self.getState()

        # Normalize the state
        '''
        Realsense:  [0, 12] => [0,1]
        LiDAR:      [0, 30] => [0,1]
        roll, pitch:[-180, 180] => [0,1]
        '''
        # scan_data = [i/30 for i in scan_data]
        realsense_data = [i/12 for i in realsense_data]
        for pa in past_action:
            realsense_data.append(pa)

        state = realsense_data + [rel_dis / diagonal_dis, (roll+180)/360, (pitch+180)/360, yaw / 360, rel_theta / 360, diff_angle / 180]
        reward = self.setReward(done, arrive)

        return np.asarray(state), reward, done, arrive

    def reset(self):
        # Reset the env #
        rospy.wait_for_service('/gazebo/delete_model')
        self.del_model('target')

        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        # Build the targetz
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        try:
            goal_urdf = open(goal_model_dir, "r").read()
            target = SpawnModel
            target.model_name = 'target'  # the same with sdf name
            target.model_xml = goal_urdf
            # Get goal position
            self.goal_position.position.x, self.goal_position.position.y = self.goal_on_law()

            self.goal(target.model_name, target.model_xml, 'namespace', self.goal_position, 'world')

            # Affine Goal Position to NavSatFix(x -> -y, y->x)
            self.goal_position.position.x = -self.goal_position.position.y
            self.goal_position.position.y = self.goal_position.position.x

        except (rospy.ServiceException) as e:
            print("/gazebo/failed to build the target")
        rospy.wait_for_service('/gazebo/unpause_physics')

        # Get sensor data
        # self.getSensor()

        self.goal_distance = self.getNavGoalDistance()
        realsense_data, rel_dis, roll, pitch, yaw, rel_theta, diff_angle, done, arrive = self.getState()
        # scan_data = [i/30 for i in scan_data]
        realsense_data = [i/12 for i in realsense_data]

        # Add Past Action
        realsense_data.append(0)
        realsense_data.append(0)

        # Normalize the state
        state = realsense_data + [rel_dis / diagonal_dis, (roll+180)/360, (pitch+180)/360, yaw / 360, rel_theta / 360, diff_angle / 180]

        return np.asarray(state)

    def goal_on_law(self):
        x_ = 0
        y_ = 0

        while True:
            x_ = random.uniform(-10.0, 10.0)
            y_ = random.uniform(-10.0, 10.0)
            
            dist1 = math.hypot(x_+0.6, y_+0.6)
            dist2 = math.hypot(x_+0.6, y_-0.6)
            dist3 = math.hypot(x_-0.6, y_-0.6)

            if (dist1 > 0.2) or (dist2 > 0.2) or (dist3 > 0.2):
                break
        return x_, y_
                
    def box_affine(self, p, threshold_affine):
        # threshold_affine = 0.2
        x, y, z = p[0], p[1], p[2]
        if (x<threshold_affine and y<threshold_affine and z<threshold_affine):
            k = threshold_affine / max(map(abs, (x,y,z)))
            x, y, z = map(lambda x: x*k, (x,y,z))
        return x, y, z
    
    def ball_affine(self, p, threshold_affine):
		# threshold_affine = 0.2
        x, y, z = p[0], p[1], p[2]
        point_dist = np.linalg.norm((x,y,z))
        if (point_dist < threshold_affine):
            k = point_dist / threshold_affine
            x, y, z = map(lambda x: x/k, (x,y,z))
        return x, y, z

    def is_outbound(self):
        x = self.nav_position[0]
        y = self.nav_position[1]
        print(x,y)
        if abs(x) > 13.5 or abs(y) > 13.5:
            return True
        return False

    def is_done(self):
        min_range = 1.2
        if abs(self.roll) > 45 or abs(self.pitch) > 45:
            print("Roll/Pitch danger")
            return True
        if min_range > min(self.scan_data) > 0 and self.is_outbound():
            print("Collision")
            return True
        return False

    def getEular(self, quaternion):
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        w = quaternion.w
        # roll
        sinr_cosp = 2.0*(w*x+y*z)
        cosr_cosp = 1-2.0*(x*x+y*y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        # pitch
        sinp = 2.0*(w*y-z*x)
        if abs(sinp) > 1:
            pitch = pi/2 if sinp > 0 else -pi/2 # Use pi/2 if out of range
        else: 
            pitch = math.asin(sinp)
        # yaw
        siny_cosp = 2.0*(w*z + x*y)
        cosy_cosp = 1-2.0*(y*y+z*z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        if yaw >= 0:
            yaw = yaw
        else:
            yaw = yaw + 360
        return roll*180/pi, pitch*180/pi, yaw*180/pi

    # def getSensor(self):
    #     realsense_data = None
    #     imu_data = None
    #     self.realsense_point = []
    #     self.roll = None
    #     self.pitch = None
    #     while realsense_data is None:
    #         try:
    #             realsense_data = rospy.wait_for_service('realsense/downsample', PointCloud2, timeout=5)
    #         except:
    #             pass
    #     while imu_data is None:
    #         try:
    #             imu_data = rospy.wait_for_service('imu/data', Imu, timeout=5)
    #         except:
    #             pass
    #     if realsense_data is not None:
    #         rs_generator = pc2.read_points(realsense_data, skip_nans=True, field_names=("x","y","z"))
    #         rs_point_length = len(list(rs_generator))

    #         self.realsense_point = list(rs_generator)
    #         self.realsense_point.extend([zero_point for _ in range(REALSENSE_MAX_POINT-rs_point_length)])

    #     if imu_data is not None:
    #         q_data = imu_data.orientation
    #         eular_data = self.getEular(q_data)
    #         self.roll = eular_data[0]
    #         self.pitch = eular_data[1]