import numpy as np

import sys
sys.path.insert(0, "/home/asalvi/code_workspace/RL_AdpEst/MachineLearning/VehicleModel/")

from BIcyclemodel_BMW import VehicleModel
import torch


class KalmanFilterWithControl:
    def __init__(self, x0, P0, Q, R, dt):
        """
        Initialize the Kalman Filter with control input.
        
        Parameters:
        x0 : np.array
            Initial state estimate [position, velocity]
        P0 : np.array
            Initial estimate uncertainty
        Q : np.array
            Process noise covariance
        R : np.array
            Measurement noise covariance
        dt : float
            Time step
        """
        self.dt = dt  # Time step
        self.x = x0  # State estimate [position, velocity]
        self.P = P0  # Estimate uncertainty
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.gT = []  #Ground truth
        self.S = None  # Ensure S exists
        self.multB = None

    def plant_dyn(self, dt):
        """
        Compute state transition (A) and control matrix (B).
        
        Parameters:
        dt : float
            Time step
        
        Returns:
        A : np.array
            State transition matrix
        B : np.array
            Control input matrix
        """
        velocity = 25
        VM = VehicleModel(velocity)
        
        #A = np.array([[1, dt], 
        #              [0, 1]])
        #B = np.array([[0.5 * dt**2], 
        #              [dt]])  

        
        A = np.eye(2) + self.dt*VM.A
        self.multB = np.array([[self.dt + 0.5*VM.A[0,0]*(self.dt**2),VM.A[0,1]*self.dt],[VM.A[1,0]*self.dt,self.dt + 0.5*VM.A[1,1]*(self.dt**2)]])
        B = self.multB*VM.B
        

        return A, B

    def predict(self, u=None):
        """
        Predict the next state and uncertainty.
        
        Parameters:
        u : np.array (optional)
            Control input (e.g., acceleration)
        """
        A, B = self.plant_dyn(self.dt)  # Get system matrices
        
        if u is not None:
            self.x = np.dot(A, self.x) + np.dot(B, u)  # Apply control input
        else:
            self.x = np.dot(A, self.x)  # Predict without control
        
        self.gT = self.x + np.dot(B,np.random.normal(0,self.Q))

        #self.P = np.dot(A, np.dot(self.P, A.T)) + np.dot(B,np.dot(self.Q,B.T))  # Update uncertainty
        multiplier = self.multB
        self.P = np.dot(A, np.dot(self.P, A.T)) + np.dot(multiplier,np.dot(self.Q,multiplier.T))  # Update uncertainty
        #print(f"P shape{self.P.shape}")

 
    def update(self, z):
        """
        Update the state estimate with a new measurement.
        
        Parameters:
        z : np.array
            New measurement (only position observed)
        """
        H = np.array([[0,1]])  # Measurement matrix (only position measured)
        #print(f"H shape{H.shape}")
        self.y = z - np.dot(H, self.x)  # Measurement residual
        #print(f"z shape{z.shape}")
        #print(f"y shape: {self.y.shape}")
        self.predy =  np.dot(H, self.x)
        self.S = np.dot(H, np.dot(self.P, H.T)) + self.R  # Residual covariance
        self.S = self.S.item() + 1e-6 #Adding an extremely small epsilon
        #print(f"S is{self.S}")
        #K = np.dot(self.P, np.dot(H.T, np.array([np.linalg.inv(self.S)])))  # Kalman Gain
        K = np.dot(self.P, np.dot(H.T, 1/self.S)) # Kalman Gain
        
        self.x = self.x + np.dot(K, self.y)  # Update state estimate
        self.P = np.dot(np.eye(len(self.P)) - np.dot(K, H), self.P)  # Update uncertainty
 