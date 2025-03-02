import gymnasium as gym
import numpy as np
from torch import nn as nn
import cv2
from scipy.linalg import solve_discrete_are
from gymnasium import Env
from gymnasium.spaces import Box
import random
import torch

import os

import sys
sys.path.insert(0, "/home/asalvi/code_workspace/RL_AdpEst/train/")

class AdaptiveRLEnv(Env):
    def __init__(self, seed):
        self.seed = seed

        self.M = 1000
        self.delta = []
        self.Q = 0.0001
        self.R = 0.0001

        # Limits and definitions on Observation and action space
        # Action space def : [Left wheel velocity (rad/s), Right wheel velocity (rad/s)]
        self.action_space = Box(low=np.array([[-1], [-1]]), high=np.array([[1], [1]]), dtype=np.float32)

        # Observation shape definition
        self.observation_space = Box(low=-100, high=100, shape=(100,1), dtype=np.float32)

        # Initial declaration of variables
        self.episode_length = 1000

    def step(self, action):

        #'LowerLimit',[0.0001;0.0001] ,'UpperLimit',[1;1])
        covariance = action
        self.Q = 0.49995*covariance[0].item() + 0.50005
        self.R = 0.49995*covariance[1].item() + 0.50005

        self.delta = self.filterModel(self.Q, self.R, self.M)
        #self.delta = np.clip(self.delta, -10, 10) 

        # Send observation for learning
        self.state = self.delta.astype(np.float32)

        kmax = max(self.delta)
        kmin = min(self.delta)
        s = (self.delta - kmin) / (kmax - kmin)
        reward = (1 - np.linalg.norm(s)) ** 2
        reward = np.float64(reward)

        # Check for reset conditions
        if self.episode_length == 0:
            done = True
        else:
            done = False

        # Update Global variables
        self.episode_length -= 1  # Update episode step counter
        print(self.episode_length)

        info = {}

        return self.state, reward, done, False, info

    def reset(self, seed=None):
        super().reset(seed=self.seed)

        self.M = 1000
        self.delta = []
        self.Q = 0.0001
        self.R = 0.0001

        self.Q = 0.49995*0 + 0.50005
        self.R = 0.49995*0 + 0.50005

        # Send observation for learning
        self.delta = self.filterModel(self.Q, self.R, self.M)
        #self.delta = np.clip(self.delta, -10, 10) 

        # Send observation for learning
        self.state = self.delta.astype(np.float32)

        self.episode_length = 1000

        info = {}

        return self.state, info

    def render(self):
        pass

   

    def filterModel(self, Q, R, M):
        dt = 0.1
        phi = torch.tensor([[1, dt], [0, 1]], dtype=torch.float32)
        B = torch.tensor([[0.5 * dt**2], [dt]], dtype=torch.float32)
        H = torch.tensor([[1, 0]], dtype=torch.float32)

        time = torch.arange(0, 100 + dt, dt)
        Q_orig = 0.0025
        R_orig = 0.01
        wk = torch.sqrt(torch.tensor(Q_orig)) * torch.randn(len(time))
        vk = torch.sqrt(torch.tensor(R_orig)) * torch.randn(len(time))

        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move tensors to the GPU
        phi = phi.to(device)
        B = B.to(device)
        H = H.to(device)
        wk = wk.to(device)
        vk = vk.to(device)
        
        # True model
        x = torch.zeros((phi.shape[0], len(time)), device=device)
        x[:, 0] = torch.ones(phi.shape[0], device=device)
        for i in range(len(time) - 1):
            x[:, i + 1] = phi @ x[:, i] + B.flatten() * wk[i]

        y = H @ x + vk

        # Initial values for state estimation
        xest_pred = torch.zeros((phi.shape[0], len(time)), device=device)
        xest_upd = torch.zeros((phi.shape[0], len(time)), device=device)
        xest_init = 0.01 * torch.ones(phi.shape[0], device=device)
        xest_upd[:, 0] = xest_init
        xest_pred[:, 0] = xest_init

        # Ensure Q_0 and R_0 are scalar values
        Q_0 = float(Q)  # Ensure Q is a scalar
        R_0 = float(R)  # Ensure R is a scalar

        # Solve the Discrete Algebraic Riccati Equation (DARE)
        P = torch.tensor(solve_discrete_are(phi.cpu().numpy().T, H.cpu().numpy().T, 
                                            Q_0 * (B @ B.T).cpu().numpy(), R_0), dtype=torch.float32, device=device)

        # Add epsilon to the denominator to avoid division by zero
        epsilon = 1e-8  # Small value to avoid division by zero
        denom = H @ P @ H.T + R_0
        W = (P @ H.T) / (denom + epsilon)  # Initial Kalman gain, avoid divide by zero

        # Debugging: Check if Kalman gain contains invalid values
        if torch.isnan(W).any() or torch.isinf(W).any():
            print("Warning: Kalman gain contains invalid values.")
            print("Denominator:", denom)
            print("Kalman gain W:", W)

        N = y.shape[1]
        W_0 = W
        nu = torch.zeros_like(y, device=device)
        mu = torch.zeros_like(y, device=device)

        # Kalman filter loop
        for i in range(len(time) - 1):
            # State propagation equation
            xest_pred[:, i + 1] = phi @ xest_upd[:, i]
            # Innovation sequence
            nu[:, i + 1] = y[:, i + 1] - H @ xest_pred[:, i + 1]
            # State update equation
            xest_upd[:, i + 1] = xest_pred[:, i + 1] + W_0 @ nu[:, i + 1]
            # Post-residual fit
            mu[:, i + 1] = y[:, i + 1] - H @ xest_upd[:, i + 1]

        # Generate covariance samples
        reduced_size = N - M  # This gives 901 in your case

        # Adjust C_est size based on reduced time dimension
        C_est = torch.zeros((vk.shape[0], vk.shape[0], M), device=device)

        M = min(100, len(time) - 1)  # Ensure we don’t exceed the time steps
        C_est = torch.zeros((M,1), device=device)  # Expecting a (1, 100) output
        # Vectorized GPU-based calculation for covariance samples
        for i in range(M):
            # Ensure both tensors have shape [1, 901] for each i
            print(f"Iteration {i}:")
            print(f"nu shape: {nu.shape}")
            print(f"nu[:, :reduced_size] shape: {nu[:, :reduced_size].shape}")
            print(f"nu[:, i:i + reduced_size] shape: {nu[:, i:i + reduced_size].shape}")

            # Adjust covariance calculation for truncated dimensions
            C_est[:, i] = (nu[:, i:i + reduced_size] @ nu[:, :reduced_size].T) / reduced_size

        # Move back to CPU if necessary
        print(C_est.shape)
        return C_est.cpu().numpy().squeeze() if device.type == "cuda" else C_est.numpy().squeeze()
    
    def plant_dyn(self, dt):
        A = np.array([[1, dt], 
                    [0, 1]])
        B = np.array([[0.5 * dt**2], 
                    [dt]])  
        return A, B




