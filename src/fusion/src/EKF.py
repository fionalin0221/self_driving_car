import numpy as np
from math import cos, sin, atan2

class ExtendedKalmanFilter:
    def __init__(self):
        # Define what state to be estimate
        # Ex.
        #   only pose -> np.array([x, y, yaw])
        #   with velocity -> np.array([x, y, yaw, vx, vy, vyaw])
        #   etc...
        self.pose = np.array([0, 0, 0])
        
        # Transition matrix
        self.A = np.identity(3)
        self.B = np.identity(3)
        
        # State covariance matrix
        self.S = np.identity(3) * 1
        
        # Observation matrix
        self.C = np.identity(3)
        
        # State transition error
        self.R = np.identity(3) * 0.01
        
        # Measurement error
        self.Q = np.identity(3) * 0.001
        self.Q[2,2] = 2 
        print("Initialize Kalman Filter")
    
    def predict(self, u):
        # Base on the Kalman Filter design in Assignment 3
        # Implement a linear or nonlinear motion model for the control input
        # Calculate Jacobian matrix of the model as self.A
        self.B = np.array([[cos(self.pose[2]),-sin(self.pose[2]), 0],
                           [sin(self.pose[2]), cos(self.pose[2]), 0],
                           [                0,                 0, 1]])
        self.pose = (self.A).dot(self.pose)+(self.B).dot(u)
        self.S = ((self.A).dot(self.S)).dot(np.transpose(self.A)) + self.R

        # raise NotImplementedError
    
        
    def update(self, z):
        # Base on the Kalman Filter design in Assignment 3
        # Implement a linear or nonlinear observation matrix for the measurement input
        # Calculate Jacobian matrix of the matrix as self.C
        
        self.K = ((self.S).dot(np.transpose(self.C))).dot(np.linalg.inv(((self.C).dot(self.S)).dot(np.transpose(self.C))+self.Q))
        self.pose = self.pose + (self.K).dot(z-(self.C).dot(self.pose))
        self.S = (np.identity(3)-(self.K).dot(self.C)).dot(self.S)

        # raise NotImplementedError
        return self.pose, self.S
    
    
    
        