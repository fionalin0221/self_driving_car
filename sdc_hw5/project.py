#!/usr/bin/env python

import rospy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2

from sensor_msgs.msg import CompressedImage, PointCloud2
import sensor_msgs.point_cloud2 as pcl2
from cv_bridge import CvBridge

pointcloud_buf = []
img_buf = []
if_subscribe_lidar = False
if_subscribe_camera = False

def read_point_from_msg(msg):

    points_list = []
    for point in pcl2.read_points(msg, skip_nans=True, field_names=("x", "y", "z", "intensity")):
        points_list.append(point)

    return np.asarray(points_list, dtype=np.float32)

def lidar_callback(msg):
    global if_subscribe_lidar
    global pointcloud_buf
    if_subscribe_lidar = True
    pointcloud = read_point_from_msg(msg)
    pointcloud_buf = pointcloud

def camera_callback(msg):
    global if_subscribe_camera
    global img_buf
    if_subscribe_camera = True
    img = CvBridge().compressed_imgmsg_to_cv2(msg)
    img_buf = img
    

if __name__ == "__main__":
    # ROS-node initialize
    rospy.init_node('project', anonymous=True)
    rate = rospy.Rate(5) # 10hz
    rospy.Subscriber("/points",                PointCloud2,      lidar_callback,  queue_size=None)
    rospy.Subscriber("/left/image/compressed", CompressedImage,  camera_callback, queue_size=None)

    # the transformation matrix from calibration.py
    fx = 698.939
    fy = 698.939
    cx = 1280/2
    cy = 720/2
    intrinsic_matrix = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]])
    # transformation_matrix = np.array([[ 0.91905587, 0.39137931,  0.04646016,  0.36238394],
    #                                   [ 0.13060528,-0.19121127, -0.97282091, -0.13055378],
    #                                   [-0.37185827, 0.9001447,  -0.22685004, -0.3316675 ],
    #                                   [ 0.          ,0.          ,0.          ,1.        ]])
    transformation_matrix = np.array([[ 0.8594, 0.5112,  -0.0049,  0.1287],
                                      [ 0.0347,-0.0680, -0.9971, -0.3280],
                                      [-0.5101, 0.8567,  -0.0762, -0.1394 ],
                                      [ 0.          ,0.          ,0.          ,1.        ]])
    # print(transformation_matrix)

    # publish the combination of lidar info and camera image

    width = 1280
    height = 720
    x = 0

    plt.ion()
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    
    while not rospy.is_shutdown():
        if if_subscribe_camera == False or if_subscribe_lidar == False:
            continue

        point_cloud_now = pointcloud_buf
        image_now = img_buf
        if isinstance(point_cloud_now,np.ndarray) == False:
            continue

 
        px_all = []
        py_all = []
        color = []
        # print(point_cloud_now.shape[0])
        inverse_trans_mx = np.linalg.inv(transformation_matrix)
        # print(inverse_trans_mx)
        for i in range(point_cloud_now.shape[0]):
            camera_point = np.array([[point_cloud_now[i,0]],[point_cloud_now[i,1]],[point_cloud_now[i,2]],[1]])
            pw = transformation_matrix @ camera_point
            # print(world_point)
            world_point = np.array([[pw[0,0]/pw[2,0]],[pw[1,0]/pw[2,0]],[1]])
            pixel_point = intrinsic_matrix @ world_point
            px = pixel_point[0,0]
            py = pixel_point[1,0]
            if px < 0 or py < 0 or px >= width or py >= height:
                continue
            # print(uv.shape)
            px_all = np.append(px_all, [px], axis = 0)
            py_all = np.append(py_all, [py], axis = 0)
            dist = np.linalg.norm(point_cloud_now[i])
            color = np.append(color, [dist])

            # print(pixel_point[0,0])
            # print(pixel_point[1,0])
        ax = fig.add_subplot()
        # print(px_all)
        plt.imshow(image_now)

        ax.set_xlim(0, 1280)
        ax.set_ylim(720, 0)

        print(px_all)
        ax.scatter(px_all, py_all, c=color, marker=',', s=10, edgecolors='none', alpha=0.7, cmap='jet')
        # ax.set_axis_off()

        plt.pause(0.1)
        plt.clf()

        rate.sleep()