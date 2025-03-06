import numpy as np
import pickle
import torch
import sys
from tqdm import tqdm

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
from VDkalman_filter_control import KalmanFilterWithControl
from VDsample_autocorrelation import SampleAutocorrelation 
from VDnis_ import NIS

from VDFourwheelmodel_Plots import fourwheel_model
from VDBIcyclemodel_BMW import VehicleModel

from VDlabelQR import LabelQR


#sys.path.insert(0, "/home/asalvi/code_workspace/RL_AdpEst/MachineLearning/")
#from kalman_filter_control import KalmanFilterWithControl
#from VehicleModel.VDlabelQR import LabelQR

class DataGenerator:
    def __init__(self, num_samples, filter_timesteps=100, n_filters=1):
        self.num_samples = num_samples
        self.filter_timesteps = filter_timesteps
        self.n_filters = n_filters

    def generate_single_sample(self):
        """Generates one data sample with true and simulated measurements."""
        # Random Q_true and R_true
        #Q_true = np.random.uniform(0, 2)
        #Here we get Q with this:
        
        mass_random = np.random.uniform(0,500)
        inertia_random = np.random.uniform(0,100)
        LbQR = LabelQR(mass_random,inertia_random)
        values = LbQR.getQ()
        Q_true = values["Q"] #Label Q
        #print(Q_true)
        R_true = np.random.uniform(0, 1) #Label R

        true_measurements = values["TrueMeasureYaw"] + np.sqrt(R_true)*np.random.normal(0,1,size=len(values["TrueMeasureYaw"]))

        plant_yawrate = values["TrueMeasureYaw"] 
        plant_beta = values["TrueMeasureBeta"] 
        plant_Xcg = values["TrueMeasureXcg"] 
        plant_Ycg = values["TrueMeasureYcg"]

        Qa_dummy = np.random.uniform(0,0.1) 
        Qb_dummy = np.random.uniform(0,0.1)
        R_dummy = np.random.uniform(0,0.3)

        # Initialize Kalman filter for ground truth measurements
        kf_true = KalmanFilterWithControl(
            x0=np.array([0, 0]), 
            P0=np.array([[1, 0], [0, 1]]), 
            Q=np.diag([Qa_dummy,Qb_dummy]), 
            R=R_dummy, 
            dt=0.01
        )

        residual_measurements = []

        for _ in range(self.filter_timesteps):
            kf_true.predict()
            kf_true.update(true_measurements[_])
            rm =  kf_true.y
            residual_measurements.append(rm)


        return {
            "true_measurements": true_measurements,
            "residual_measurements": residual_measurements,
            "Qa_true": Q_true[0,0],
            "Qb_true": Q_true[1,1],
            "R_true": R_true,
            "plant_yawrate" : plant_yawrate,
            "plant_beta" : plant_beta,
            "plant_Xcg" : plant_Xcg,
            "plant_Ycg" : plant_Ycg,
            "Qa_dummy" : Qa_dummy,
            "Qb_dummy" : Qb_dummy,
            "R_dummy" : R_dummy,
        }

    def generate_dataset(self, save_path="vehicle_datasetB.pkl"):
        """Generates and saves the dataset."""
        dataset = []
        
        for _ in tqdm(range(self.num_samples), desc="Generating Data"):
            sample = self.generate_single_sample()
            #print(len(sample["true_measurements"]))
            #print(len(sample["residual_measurements"]))
            #print((sample["Qa_true"]))
            #print((sample["Qb_true"]))
            #print((sample["R_true"]))
            dataset.append(sample)
            #del sample

        # Save as pickle file
        with open(save_path, "wb") as f:
            pickle.dump(dataset, f)

        print(f"Dataset saved to {save_path}")

if __name__ == "__main__":
    generator = DataGenerator(num_samples=50000)
    generator.generate_dataset()
