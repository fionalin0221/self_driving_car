#!/usr/bin/env python3

import rospy
import os
import numpy as np
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import PoseWithCovarianceStamped as Pose
from nav_msgs.msg import Odometry
from math import cos, sin, atan2, sqrt
from matplotlib import pyplot as plt

from EKF import ExtendedKalmanFilter

class Fusion:
    def __init__(self):
        rospy.Subscriber('/gps', Pose, self.gpsCallback)
        rospy.Subscriber('/radar_odometry', Odometry, self.odometryCallback)
        rospy.Subscriber('/gt_odom', Odometry, self.gtCallback)
        rospy.on_shutdown(self.shutdown)
        self.posePub = rospy.Publisher('/pred', Odometry, queue_size = 10)
        self.EKF = None
        
        self.gt_list = [[], []]
        self.est_list = [[], []]
        self.initial = False

    def shutdown(self):
        print("shuting down fusion.py")

    def predictPublish(self):
        
        predPose = Odometry()
        predPose.header.frame_id = 'origin'
        # change to the state x and state y from EKF
        predPose.pose.pose.position.x = ???
        predPose.pose.pose.position.y = ???
        
        # Change to the state yaw from EKF
        quaternion = quaternion_from_euler(0, 0, ???)
        predPose.pose.pose.orientation.x = quaternion[0]
        predPose.pose.pose.orientation.y = quaternion[1]
        predPose.pose.pose.orientation.z = quaternion[2]
        predPose.pose.pose.orientation.w = quaternion[3]
        
        # Change to the covariance matrix of [x, y, yaw] from EKF
        predPose.pose.covariance = [???, ???, 0, 0, 0, ???,
                                    ???, ???, 0, 0, 0, ???,
                                    0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0,
                                    ???, ???, 0, 0, 0, ???]
                                    
        self.posePub.publish(predPose)
    
    def odometryCallback(self, data):
        odom_x = data.pose.pose.position.x
        odom_y = data.pose.pose.position.y
        odom_quaternion = [
            data.pose.pose.orientation.x, data.pose.pose.orientation.y,
            data.pose.pose.orientation.z, data.pose.pose.orientation.w
        ]
        _, _, odom_yaw = euler_from_quaternion(odom_quaternion)
        odom_covariance = np.array(data.pose.covariance).reshape(6, 6)
        
        # Design the control of EKF state from radar odometry data
        # The data is in global frame, you may need to find a way to convert it into local frame
        # Ex. 
        #     Find differnence between 2 odometry data 
        #         -> diff_x = ???
        #         -> diff_y = ???
        #         -> diff_yaw = ???
        #     Calculate transformation matrix between 2 odometry data
        #         -> transformation = last_odom_pose^-1 * current_odom_pose
        #     etc.
        control = ???
        
        if not self.initial:
            self.initial = True
            self.EKF = ExtendedKalmanFilter(odom_x, odom_y, odom_yaw)
        else:
            # Update error covriance
            self.EKF.R = ???
            self.EKF.predict_EKF(u = control)

        self.predictPublish()
        
    def gpsCallback(self, data):
        gps_x = data.pose.pose.position.x
        gps_y = data.pose.pose.position.y
        gps_covariance = np.array(data.pose.covariance).reshape(6, 6)
        
        # Design the measurement of EKF state from GPS data
        # Ex. 
        #     Use GPS directly
        #     Find a approximate yaw
        #     etc.
        measurement = ???
        
        if not self.initial:
            self.initial = ExtendedKalmanFilter(gps_x, gps_y)
            self.initial = True
        else:
            # Update error covriance
            self.EKF.Q = ???
            self.EKF.update(z = measurement)
            
        self.predictPublish()
        return
    
    def gtCallback(self, data):
        self.gt_list[0].append(data.pose.pose.position.x)
        self.gt_list[1].append(data.pose.pose.position.y)
        if self.EKF is not None:
            # Change to the state x and state y from EKF
            self.est_list[0].append(???) 
            self.est_list[1].append(???)
        return

    def plot_path(self):
        plt.figure(figsize=(10, 8))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.plot(self.gt_list[0], self.gt_list[1], alpha=0.25, linewidth=8, label='Groundtruth path')
        plt.plot(self.est_list[0], self.est_list[1], alpha=0.5, linewidth=3, label='Estimation path')
        plt.title("KF fusion odometry result comparison")
        plt.legend()
        if not os.path.exists("/root/catkin_ws/result"):
            print("not exist")
            os.mkdir("/root/catkin_ws/result")
        plt.savefig("/root/catkin_ws/result/result.png")
        plt.show()
        return
    
if __name__ == '__main__':
    rospy.init_node('kf', anonymous=True)
    fusion = Fusion()
    rospy.spin()
    fusion.plot_path()