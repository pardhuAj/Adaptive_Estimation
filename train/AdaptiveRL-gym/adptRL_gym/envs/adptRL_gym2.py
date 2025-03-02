import gymnasium as gym
import numpy as np
from torch import nn as nn
import cv2
from scipy.linalg import solve_discrete_are
from gymnasium import Env
from gymnasium.spaces import Box
import random
import torch
import sys
sys.path.insert(0, "/home/asalvi/code_workspace/RL_AdpEst/train/")
sys.path.insert(0, "/home/asalvi/code_workspace/RL_AdpEst/train/AdaptiveRL-gym/adptRL_gym/envs/")
from kalman_filter_control import KalmanFilterWithControl  # Import the class
import scipy.stats as stats
from scipy.linalg import solve_discrete_are

import os



class AdaptiveRLEnv(Env):
    def __init__(self, seed):
        self.seed = seed

        self.N = 100 # We define/ Play with batch size
        self.M = self.N /2

        #Initializations
        self.dt = 0.1
        self.Q = 0.0001 
        self.R = 0.0001
        self.GT_Q = []
        self.GT_R = []
        self.x0 = np.array([0, 0])
        self.P0 = np.array([[1, 0], [0, 1]])
        self.u = np.array([0.1])  # Assume constant acceleration of 0.1 m/s²

        self.obsA = [] #sliding window observation Cache A
        self.obsB = [] #sliding window observation Cache A
        self.S_ev = [] 

        self.refMes = []

        # Limits and definitions on Observation and action space
        # Action space def : [Left wheel velocity (rad/s), Right wheel velocity (rad/s)]
        self.action_space = Box(low=np.array([[-1],[-1]]), high=np.array([[1],[1]]), dtype=np.float32)

        # Observation shape definition
        self.observation_space = Box(low=-100, high=100, shape=(2,self.N), dtype=np.float32)

        # Initial declaration of variables
        self.episode_length = 1000
        self.step_no = 0

    def step(self, action):

        self.Q = action[0] + 1
        self.R = action[1] + 1

        #print(f"self.Q{self.Q}")
        #print(f"self.R{self.R}")


        #self.kf.predict()
        #measurement = self.kf.gT[0] + self.R
        #measurement = self.refMes[self.step_no]
        #self.kf.update(measurement)

        for i in range(self.N):
            self.kf.predict()
            measurement = self.refMes[self.step_no]
            self.kf.update(measurement)
            
            # Append new NumPy arrays to the observation lists
            self.obsA = np.append(self.obsA , measurement)  
            self.obsB = np.append(self.obsB , self.kf.predy)

            self.step_no += 1

        # Ensure the lists maintain only the last 10 elements (sliding window)
        #self.obsA = self.obsA[-10:]  # Keep only the last 10 arrays
        #self.obsB = self.obsB[-10:]  # Keep only the last 10 arrays
            

            ### >>>>>>>>>>>

        

            ### <<<<<<<<<<<<

        # Convert the last 10 elements into a NumPy array
        observations = np.array([self.obsA[-self.N:], self.obsB[-self.N:]], dtype=np.float32)


        # Send observation for learning
        self.state = observations.astype(np.float32)

        self.S_ev = np.append(self.S_ev,self.kf.S)
        self.seq = self.S_ev[-self.N:]
        
        reward = self.get_reward()

        # Check for reset conditions
        if self.episode_length == 0:
            done = True
        else:
            done = False

        # Update Global variables
        self.episode_length -= 1  # Update episode step counter
        #self.step_no +=1
        #print(self.episode_length)

        info = {}

        return self.state, reward, done, False, info

    def reset(self, seed=None):
        super().reset(seed = seed)

        self.episode_length = 1000
        self.step_no +=0

        #Initializations
        self.dt = 0.1
        self.Q = np.random.uniform(0,2) #We generate Q randomly between zero to 2 at begining of each run
        self.GT_Q = self.Q
        #print(f"GT_Q: {self.GT_Q}")
        self.R = np.random.uniform(0,2)   #We generate R randomly between zero to 2 at begining of each run
        self.GT_R = self.R
        #print(f"GT_R: {self.GT_R}")
        self.x0 = np.array([0, 0])
        self.P0 = np.array([[1, 0], [0, 1]])
        self.u = np.array([0.1])  # Assume constant acceleration of 0.1 m/s²

        # Initialize Kalman Filter 
        self.kf = KalmanFilterWithControl(self.x0, self.P0, self.GT_Q, self.GT_R, self.dt)

        H = np.array([1,0])

        for i in range(self.N*self.episode_length):
            self.kf.predict()
            measurement = np.dot(H,self.kf.gT) + np.random.normal(0,self.R)
            #print(f"gt shape{H*self.kf.gT[0]}")
            #print(len(self.kf.gT[[0]]))
            #print(measurement)
            self.kf.update(measurement[0])
            self.refMes.append(measurement[0])

        #print(len(self.refMes))

        #del self.kf

        # Initialize Kalman Filter 
        self.Q = np.random.uniform(0,2) #Only for random initialization for first observation
        self.R = np.random.uniform(0,2)   #Only for random initialization for first observation
        self.kf = KalmanFilterWithControl(self.x0, self.P0, self.Q, self.R, self.dt)
        self.kf.predict()
        measurement = np.dot(H,self.kf.gT) + np.random.normal(0,self.R)
        self.kf.update(measurement[0])

        self.S_ev = self.kf.S*np.ones((self.N))

        obs_ = measurement[0] * np.ones(self.N)
        self.obsA = obs_  # Initialize with one element
        #print(len(self.obsA))
        obs_ = self.kf.predy * np.ones(self.N)
        self.obsB = obs_  # Initialize with one element
        #print(len(self.obsB))

        # Correct indexing (returning the last available observation)
        observations = np.array([self.obsA,self.obsB], dtype=np.float32)

        self.state = observations  # Ensure self.state is correctly formatted


        # Send observation for learning
        self.state = observations.astype(np.float32)

        

        info = {}

        return self.state, info

    def render(self):
        pass
    
    def get_reward(self):
        """
        Compute reward based on autocorrelation-like function.

        self.S_ev: 1x10 vector from main function
        self.N, self.M: Parameters defining vector size

        Returns:
        C_est: Estimated C values
        """
        
        # Initialize arrays correctly

        M = int(self.M)
        N = int(self.N)
        k = np.zeros(N-M)  # Create an array of zeros with integer type

        '''
        C_est = np.zeros(M)  # Ensure self.M is converted to an integer

        # Compute C_0
        C_0 = (np.sum(self.seq[:len(self.seq) // 2] ** 2)) / (N-M)
        #C_0 = (self.seq[0]*self.seq[0])/(N-M)
        #print(C_0)

        # Compute C_est[i] values
        for i in range(1, M):  # Fix syntax error in range
            for j in range(1, N- M):  # Fix syntax error in range
                k[j - 1] = self.seq[j] * self.seq[j + i]  # Fix indexing

            C_est[i] = np.sum(k) / (N-M)  # Compute the final C_est value
        '''


        # Define system matrices
        #A = np.array([[0.8, 1], [-0.4, 0]])   # State transition matrix
        dt = 0.1
        A = np.array([[1, dt], 
                      [0, 1]])
        B = np.array([[0.5 * dt**2], 
                      [dt]])  
        H = np.array([[1, 0]])                # Observation matrix

        tp = np.dot(B,np.array([np.dot(self.Q,B.T)]))

        epsilon = 1e-3

        # Solve Discrete Algebraic Riccati Equation (DARE)
        P = solve_discrete_are(A.T, H.T, tp , self.R)
        S = np.dot(H, np.dot(P, H.T)) + self.R  # Residual covariance
        S = S + epsilon
        W = np.dot(P, np.dot(H.T, 1/S)) # Kalman Gain

        C_0 = np.dot(H,np.dot(P,H.T)) + self.R
        C_0 = C_0 + epsilon

        # Print the steady-state error covariance matrix
        #print("Steady-State Error Covariance (P):")
        #print(P)

        #E = np.diag(C_0)**(-0.5)  # Equivalent to E = diag(C0)^(-1/2)
        E = 1/np.sqrt(np.diag(C_0))
        E = np.array([E])
        #print(E.shape)

        # Compute X
        Psi = P @ H.T
        X = Psi - W @ C_0
        #print(X.shape)

        # Initialize summation term
        sum_term = np.zeros_like(X @ E @ E @ X.T)
        #sum_term = np.zeros_like(np.dot(X,np.dot(E,np.dot(E,X.T))))

        # Compute the summation in J
        for i in range(1, M):  # i starts from 1 to M-1
            Phi_i = H @ np.linalg.matrix_power(A, i-1) @ A  # Compute Phi(i)
            Theta_i = Phi_i.T @ E @ E @ Phi_i  # Compute Theta(i)
            sum_term += Theta_i @ X @ E @ E @ X.T

        # Compute objective function J
        J = 0.5 * np.trace(sum_term)


        """
        Compute the objective function J based on the given covariance estimates.

        Parameters:
            C_est (list or numpy.ndarray): List or array of estimated covariance matrices C_hat(i), where
                                        C_est[0] corresponds to C_hat(0) and is used for normalization.

        Returns:
            float: The computed objective function value J.
        

        M = len(C_est)  # Number of covariance matrices

        # Ensure C_0 is treated as a diagonal matrix
        if np.isscalar(C_0):  # If C_0 is a single value (float/int)
            C_0_diag_matrix = np.array([[C_0]])  # Convert to a 2D matrix
        else:
            C_0_diag_matrix = np.diag(C_0)  # Convert to diagonal if already a vector

        C_0_diag_inv_sqrt = np.linalg.inv(C_0_diag_matrix) ** 0.5  # Compute [diag(Ĉ(0))]^{-1/2}
        C_0_diag_inv = np.linalg.inv(C_0_diag_matrix)  # Compute [diag(Ĉ(0))]^{-1}

        J_sum = 0  # Initialize summation

        for i in range(1, M):  # Summation from i = 1 to M-1
            C_i = np.atleast_2d(C_est[i])  # Ensure C_i is a 2D matrix
            term1 = C_0_diag_inv_sqrt @ C_i.T  # [diag(Ĉ(0))]^{-1/2} Ĉ(i)^T
            term2 = C_0_diag_inv @ C_i @ C_0_diag_inv_sqrt  # [diag(Ĉ(0))]^{-1} Ĉ(i) [diag(Ĉ(0))]^{-1/2]
            J_sum += np.trace(term1 @ term2)  # Compute trace of the product

        J_ = 0.5 * J_sum  # Final computation

        """

        """
        Compute the time-average normalized innovation squared (NIS) statistic.

        Parameters:
            innovations (numpy.ndarray): A (K, n) matrix of innovation values where K is the number of time steps
                                        and n is the dimension of the innovation vector.
            S_matrices (numpy.ndarray): A (K, n, n) array representing the sequence of covariance matrices S(k).

        Returns:
            float: The computed time-average normalized innovation squared statistic.
        """
        #print(self.seq.shape)
        # Print the shape of self.seq for debugging
        #print(f"Shape of self.seq: {self.seq.shape}")

        K = self.seq.shape[0]  # Number of time steps
        n = self.seq.shape[1] if len(self.seq.shape) > 1 else 1  # Get the innovation dimension

        # Ensure self.seq is a 2D matrix
        self.seq = np.atleast_2d(self.seq).T  # Convert (10,) → (10, 1) to enforce column vector

        K = self.seq.shape[0]  # Number of time steps
        n = self.seq.shape[1]  # Innovation dimension

        # Initialize the sum
        NIS_sum = 0

        for k in range(K):
            nu_k = np.atleast_2d(self.seq[k])  # Ensure nu_k is at least 2D
            S_k = np.atleast_2d(self.seq[k])   # Ensure S_k is at least 2D

            # Convert S_k to a square diagonal matrix if needed
            if S_k.shape[0] != S_k.shape[1]:
                S_k = np.diag(S_k.flatten())  # Convert to a square matrix

            # Ensure S_k is invertible (add small regularization if needed)
            if np.linalg.cond(S_k) > 1e10:
                S_k += np.eye(S_k.shape[0]) * 1e-6  # Small regularization

            # Ensure nu_k is a column vector for multiplication compatibility
            nu_k = nu_k.reshape(-1, 1)  # Convert (10,) → (10,1)

            # Compute the normalized innovation squared (NIS) term
            NIS_sum += float(nu_k.T @ np.linalg.inv(S_k) @ nu_k)  # Force scalar output

        # Compute the time-averaged NIS
        NIS = (1 / K) * NIS_sum



        confidence = 0.99
        df = 1  # Change this for different measurement sizes

        alpha = 1 - confidence  # Significance level (1% for 99% confidence)

        # Get critical values from chi-square distribution
        lower_bound = stats.chi2.ppf(alpha / 2, df)  # 0.5% left tail
        upper_bound = stats.chi2.ppf(1 - alpha / 2, df)  # 99.5% right tail

        #dis_lb = abs(NIS - lower_bound)
        #dis_ub = abs(NIS - upper_bound)

        # Count number of NIS values within bounds
        within_bounds = np.logical_and(NIS >= lower_bound, NIS <= upper_bound)
        fraction_within_bounds = np.mean(within_bounds)  # Fraction of samples in range

        # Define Reward:
        # +1 for each sample within bounds, large penalty (-10) if fraction is very low
        reward = 10 * fraction_within_bounds - 10 * (1 - fraction_within_bounds)

        #bound_distance = max(dis_lb, dis_ub)


        #Rew = (reward - (1*abs(J)))**2
        Rew = -1*abs(J)

        return Rew
    

'''

from stable_baselines3.common.env_checker import check_env

if __name__ == "__main__":
    # Create an instance of the environment
    env = AdaptiveRLEnv(seed=42)  # Provide a seed for reproducibility

    # Check if the environment is valid for Stable-Baselines3
    check_env(env, warn=True)

    print("Environment validation complete. No critical issues found.")

'''
        


        







