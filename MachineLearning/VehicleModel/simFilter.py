import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt

import sys
sys.path.insert(0, "/home/asalvi/code_workspace/RL_AdpEst/MachineLearning/VehicleModel/")
from kalman_filter_control import KalmanFilterWithControl
from sample_autocorrelation import SampleAutocorrelation 
from nis_ import NIS


class SimFilter():
    def __init__(self):
        self.filter_timesteps = 10000

    def simKF(self):

        # Random Q_true and R_true
        Q_true = np.random.uniform(0, 2)
        Q_true =np.array([[0.06128198, 0],[0,0.44981162]])
        R_true = np.random.uniform(0, 1)

        # Initialize Kalman filter for ground truth measurements
        kf_true = KalmanFilterWithControl(
            x0=np.array([0, 0]), 
            P0=np.array([[1, 0], [0, 1]]), 
            Q=Q_true, 
            R=R_true, 
            dt=1e-3
        )

        true_measurements = []
        for _ in range(self.filter_timesteps):
            kf_true.predict()
            measurement = np.random.normal(0, R_true)  # Noisy observation
            kf_true.update(measurement)
            true_measurements.append(measurement)

        self.tm = true_measurements
        self.plot_tm()
    
    def plot_tm(self):

        plt.figure()
        plt.plot(self.tm, label="R_true", linestyle="dashed", color="blue", alpha=0.7)
        plt.xlabel("Sample")
        plt.ylabel("True Measurement")
        plt.legend()
        plt.title("True measurement for Validation")
        plt.grid()
        plt.show()

if __name__ == "__main__":

    SM = SimFilter()
    SM.simKF()