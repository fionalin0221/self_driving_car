import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2

def click_points_2D(image):
  
  # Setup matplotlib GUI
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_title('Select 2D Image Points')
  ax.set_axis_off()
  ax.imshow(image)

  # Pick points
  picked, corners = [], []
  def onclick(event):

    # Only clicks inside this axis are valid
    if fig.canvas.cursor().shape() != 0: 
      return

    x = event.xdata
    y = event.ydata
    if (x is None) or (y is None):
      return

    # Display the picked point
    picked.append((x, y))
    corners.append((x, y))
    print(str(picked[-1]))

    if len(picked) > 1:

      # Draw the line
      temp = np.array(picked)
      ax.plot(temp[:, 0], temp[:, 1])
      ax.figure.canvas.draw_idle()

      # Reset list for future pick events
      del picked[0]

  # Display GUI
  fig.canvas.mpl_connect('button_press_event', onclick)
  plt.show()
  
  if len(corners) > 1: del corners[-1]
  print()
  return np.array(corners)


def click_points_3D(points):

  # Select points within a specific range
  x_min, x_max = -2.0, 0
  y_min, y_max =  0.0, 2
  z_min, z_max = -1.0, 1
  inrange = np.where((points[:, 0] > x_min) & (points[:, 0] < x_max) &
                     (points[:, 1] > y_min) & (points[:, 1] < y_max) &
                     (points[:, 2] > z_min) & (points[:, 2] < z_max))
  points = points[inrange[0]]

  # Color map for the points
  cmap = matplotlib.colormaps['hsv']
  colors = cmap(points[:, -1] / np.max(points[:, -1]))

  # Setup matplotlib GUI
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.set_title('Select 3D LiDAR Points')
  ax.set_axis_off()
  ax.set_facecolor((0, 0, 0))
  ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=20, picker=5)

  # Equalize display aspect ratio for all axes
  max_range = (np.array([points[:, 0].max() - points[:, 0].min(), 
                         points[:, 1].max() - points[:, 1].min(),
                         points[:, 2].max() - points[:, 2].min()]).max() / 2.0)
  mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
  mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
  mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
  ax.set_xlim(mid_x - max_range, mid_x + max_range)
  ax.set_ylim(mid_y - max_range, mid_y + max_range)
  ax.set_zlim(mid_z - max_range, mid_z + max_range)

  # Pick points
  picked, corners = [], []
  def onpick(event):
    ind = event.ind[0]
    x, y, z = event.artist._offsets3d

    # Ignore if same point selected again
    if picked and (x[ind] == picked[-1][0] and y[ind] == picked[-1][1] and z[ind] == picked[-1][2]):
      return
    
    # Only clicks inside this axis are valid
    if fig.canvas.cursor().shape() != 0: 
      return
    
    # Display picked point
    picked.append((x[ind], y[ind], z[ind]))
    corners.append((x[ind], y[ind], z[ind]))
    print(str(picked[-1]))
    
    if len(picked) > 1:

      # Draw the line
      temp = np.array(picked)
      ax.plot(temp[:, 0], temp[:, 1], temp[:, 2])
      ax.figure.canvas.draw_idle()

      # Reset list for future pick events
      del picked[0]

  # Display GUI
  fig.canvas.mpl_connect('pick_event', onpick)
  plt.show()

  if len(corners) > 1: del corners[-1]
  print()
  return np.array(corners)

def main():
  # img_2D = cv2.imread('/root/catkin_ws/sdc_hw5/data/camera/1518069838279552217.jpg')
  # click_points_2D(img_2D)
  # img_3D = cv2.imread('/root/catkin_ws/sdc_hw5/data/lidar/1518069838220356412.npy')
  # click_points_3D(img_3D)

  array_2d = np.array([[457.344,387.690],
                       [615.138,384.741],
                       [616.612,502.085],
                       [455.659,523.901]])
  array_3d = np.array([[-0.983,0.835,-0.469],
                       [-0.859,1.096,-0.471],
                       [-0.864,1.109,-0.706],
                       [-0.975,0.835,-0.469]])

  fx = 698.939
  fy = 698.939
  cx = 128.0/2
  cy = 720.0/2
  intr_mx = np.array([[fx,0.0,cx],[0.0,fy,cy],[0.0,0.0,1.0]])
  distCoe = np.array([0.0,0.0,0.0,0.0,0.0])

  retval,rvec,tvec = cv2.solvePnP(array_3d, array_2d, intr_mx, distCoe, 0, flags=1)
  R_mx,_ = cv2.Rodrigues(rvec)
  print(R_mx)
  print(tvec)
  # print(tvec.shape)
  trans_mx = np.array([[R_mx[0,0],R_mx[0,1],R_mx[0,2],tvec[0,0]],
                       [R_mx[1,0],R_mx[1,1],R_mx[1,2],tvec[1,0]],
                       [R_mx[2,0],R_mx[2,1],R_mx[2,2],tvec[2,0]],
                       [0,0,0,1]])
  print(trans_mx)
  inverse_trans_mx = np.linalg.inv(trans_mx)
  print(inverse_trans_mx)
  # inv_R = np.linalg.inv(R_mx)
  # inv_t = - inv_R @ tvec
  # print(inv_R)
  # print(inv_t)
  camera_point = np.array([[0],[0],[0],[1]])
  pw = inverse_trans_mx @ camera_point
  # print(world_point)
  world_point = np.array([[pw[0,0]/pw[2,0]],[pw[1,0]/pw[2,0]],[1]])
  pixel_point = intr_mx @ world_point
  print(pixel_point[0,0])
  print(pixel_point[1,0])

if __name__ == "__main__":
  main()