import numpy as np

class KalmanFilter2D:
    def __init__(self, x0, P0, F, H, Q, R):
        """
        Initialize the 2D Kalman Filter.
        
        Parameters:
        x0 : np.array
            Initial state estimate [position, velocity]
        P0 : np.array
            Initial estimate uncertainty
        F : np.array
            State transition model
        H : np.array
            Measurement model
        Q : np.array
            Process noise covariance
        R : np.array
            Measurement noise covariance
        """
        self.x = x0  # State estimate [position, velocity]
        self.P = P0  # Estimate uncertainty
        self.F = F  # State transition model
        self.H = H  # Measurement model
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance

    def predict(self):
        """ Predict the next state and uncertainty. """
        self.x = np.dot(self.F, self.x)  # State prediction
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q  # Uncertainty prediction
    
    def update(self, z):
        """
        Update the state estimate with a new measurement.
        
        Parameters:
        z : np.array
            New measurement (only position observed)
        """
        self.y = z - np.dot(self.H, self.x)  # Measurement residual (innovation)
        self.S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R  # Residual covariance
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(self.S)))  # Kalman Gain
        
        self.x = self.x + np.dot(K, self.y)  # Update state estimate
        self.P = np.dot(np.eye(len(self.P)) - np.dot(K, self.H), self.P)  # Update uncertainty