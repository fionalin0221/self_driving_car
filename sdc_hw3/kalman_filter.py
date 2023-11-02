import numpy as np

class KalmanFilter:
    def __init__(self, x=0, y=0, yaw=0):
        # State [x, y, yaw]
        self.state = np.array([x, y, yaw])
        
        # Transition matrix
        self.A = np.identity(3)
        self.B = np.identity(3)
        
        # State covariance matrix
        self.S = np.identity(3) * 1
        
        # Observation matrix
        self.C = np.array([[1,0,0],[0,1,0]])
        
        # State transition error
        self.R = np.array([[1,0,0],[0,1,0],[0,0,1]])
        
        # Measurement error
        self.Q = np.array([[100,0],[0,100]])

    def predict(self, u):
        # raise NotImplementedError
        self.state = (self.A).dot(self.state)+(self.B).dot(u)
        self.S = ((self.A).dot(self.S)).dot(np.transpose(self.A)) + self.R

    def update(self, z):
        # raise NotImplementedError
        self.K = ((self.S).dot(np.transpose(self.C))).dot(np.linalg.inv(((self.C).dot(self.S)).dot(np.transpose(self.C))+self.Q))
        self.state = self.state + (self.K).dot(z-(self.C).dot(self.state))
        self.S = (np.identity(3)-(self.K).dot(self.C)).dot(self.S)
        return self.state, self.S
