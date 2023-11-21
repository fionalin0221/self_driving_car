import rospy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2

from sensor_msgs.msg import CompressedImage, PointCloud2  #大函式庫中import小函式庫
import sensor_msgs.point_cloud2 as pcl2
from cv_bridge import CvBridge

pointcloud_buf = []
img_buf = []
if_subscribe_lidar = False
if_subscribe_camera = False

def read_point_from_msg(msg):  #將ros的msg資訊轉成numpy資訊

    points_list = []
    for point in pcl2.read_points(msg, skip_nans=True, field_names=("x", "y", "z", "intensity")):
        points_list.append(point)

    return np.asarray(points_list, dtype=np.float32)

def lidar_callback(msg):
    global if_subscribe_lidar
    global pointcloud_buf
    pointcloud = read_point_from_msg(msg)
    if_subscribe_lidar = True   #進入callback時訂閱lider的資訊
    pointcloud_buf = pointcloud #把資訊存成全域變數

def camera_callback(msg):
    global if_subscribe_camera
    global img_buf
    img = CvBridge().compressed_imgmsg_to_cv2(msg)
    if_subscribe_camera = True
    img_buf = img
    

if __name__ == "__main__":
    # ROS-node initialize
    rospy.init_node('project', anonymous=True)
    rate = rospy.Rate(5) # 5hz
    rospy.Subscriber("/points",                PointCloud2,      lidar_callback,  queue_size=None)  #訂閱lider pointcloud資訊
    rospy.Subscriber("/left/image/compressed", CompressedImage,  camera_callback, queue_size=None)  #訂閱camera image資訊

    # the transformation matrix from calibration.py
    fx = 698.939
    fy = 698.939
    cx = 1280/2
    cy = 720/2
    intrinsic_matrix = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]])

    transformation_matrix = np.array([[ 0.86179114,  0.50722155, -0.00650583,  0.13298259],
                                      [ 0.02940184, -0.06275055, -0.99759606, -0.33730762],
                                      [ -0.50641047,  0.85952816, -0.06899112, -0.12735807],
                                      [ 0,           0,           0,           1        ]])


    # transformation_matrix = np.array([[ 0.85942994, 0.51122949,  -0.00495944,  0.12869505],
    #                                   [ 0.03469834,-0.0680042,  -0.99708147, -0.32801881],
    #                                   [-0.51007471, 0.85674958,  -0.07618336, -0.13943649],
    #                                   [ 0     , 0.    ,   0.      ,1.    ]])                                  


    # publish the combination of lidar info and camera image

    width = 1280
    height = 720
    x = 0

    plt.ion() #使matplotlib的顯示模式轉換為交互（interactive）模式
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)  #figsize:畫布長寬尺寸，單位為「英吋」， dpi:畫布分辨率，表示一英吋裡有幾個像素
    
    while not rospy.is_shutdown():
        # if if_subscribe_camera == False or if_subscribe_lidar == False:  #如果還沒有第一個資訊，先不要做計算
        #     continue

        point_cloud_now = pointcloud_buf.copy()  #避免上面的算太快，下面還沒算完，下面計算要用同一個值
        image_now = img_buf.copy()
        if isinstance(point_cloud_now,np.ndarray) == False:  #確認丟進來的是np.array
            continue

        # print(point_cloud_now.shape)
 
        px_all = []
        py_all = []
        color = []

        for i in range(point_cloud_now.shape[0]):  #每一時刻所有pointcloud，此處shape[0]是點個數，shape[1]是每個點裡面資訊的個數
            # if i % 3 != 0:
            #     continue
            lidar_point = np.array([[point_cloud_now[i,0]],[point_cloud_now[i,1]],[point_cloud_now[i,2]],[1]])  #導入lider位置資訊
            pc = transformation_matrix @ lidar_point                           #座標轉換
            world_point = np.array([[pc[0,0]/pc[2,0]],[pc[1,0]/pc[2,0]],[1]])
            pixel_point = intrinsic_matrix @ world_point
            px = pixel_point[0,0]
            py = pixel_point[1,0]
            if px < 0 or py < 0 or px >= width or py >= height:  #
                continue
            # print(uv.shape)
            px_all = np.append(px_all, [px], axis = 0)  #將此點轉換好的座標加入整個轉換好的座標組中
            py_all = np.append(py_all, [py], axis = 0)
            vec = np.array([point_cloud_now[i,0],point_cloud_now[i,1],point_cloud_now[i,2]])
            # print(vec)
            dist = np.linalg.norm(vec)  #算距離
            # if(dist < 0):
            #     print(dist)
            color = np.append(color, [dist])  #顏色用距離遠近表示

        ax = fig.add_subplot()
        # print(px_all)
        plt.imshow(image_now)

        ax.set_xlim(0, 1280)
        ax.set_ylim(720, 0)

        # print(px_all)
        ax.scatter(px_all, py_all, c=color, marker=',', s=10, edgecolors='none', alpha=0.7, cmap='jet')
        # ax.set_axis_off()

        plt.pause(0.01)
        plt.clf()

        rate.sleep()