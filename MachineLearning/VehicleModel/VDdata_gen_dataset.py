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

#import sys
#sys.path.insert(0, "/home/asalvi/code_workspace/RL_AdpEst/MachineLearning/VehicleModel/")
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
        
        #mass_random = np.random.uniform(0,500)
        #inertia_random = np.random.uniform(0,100)

        start = np.random.randint(1,1500)
        validate = 0
        LbQR = LabelQR(start, validate)
        values = LbQR.getQ()
        Q_true = values["Q"] #Label Q
        #print(Q_true)
        R_true = np.random.uniform(0, 0.1) #Label R

        k = values["k"] #from where to begin the traning sample

        #We generate true measurement by adding Q and R to the non-varying bicyle model
        true_measurements = values["TrueMeasureYawRate"] + np.sqrt(Q_true[1,1])*np.random.normal(0,1,size=len(values["TrueMeasureYawRate"])) + np.sqrt(R_true)*np.random.normal(0,1,size=len(values["TrueMeasureYawRate"]))
        #print(true_measurements.shape)

        plant_yawrate = values["TrueYawRate"] 
        plant_beta = values["TrueBeta"] 
        plant_Xcg = values["TrueXcg"] 
        plant_Ycg = values["TrueYcg"]

        Qa_dummy = np.random.uniform(0,0.1) 
        Qb_dummy = np.random.uniform(0,0.1)
        R_dummy = np.random.uniform(0,0.1)

        # Initialize Kalman filter for ground truth measurements
        kf_true = KalmanFilterWithControl(
            x0=np.array([0, 0]), 
            P0=np.array([[1, 0], [0, 1]]), 
            Q=np.diag([Qa_dummy,Qb_dummy]), 
            R=R_dummy, 
            dt=0.01
        )

        residual_measurements = []

        for _ in range(len(plant_yawrate)):
            kf_true.predict()
            kf_true.update(plant_yawrate[_])
            rm =  kf_true.y #get innovation sequence
            residual_measurements.append(rm.item())
        
        #print(len(residual_measurements))


        return {
            "true_measurements": true_measurements, #1,100
            "residual_measurements": residual_measurements[k:k+100], #1,100
            "Qa_true": Q_true[0,0], #1,1
            "Qb_true": Q_true[1,1], #1,1
            "R_true": R_true, #1,1
            "plant_yawrate" : plant_yawrate, #1,length of simulation time
            "plant_beta" : plant_beta, #1,length of simulation time
            "plant_Xcg" : plant_Xcg, #1,length of simulation time
            "plant_Ycg" : plant_Ycg, #1,length of simulation time
            "Qa_dummy" : Qa_dummy, #1,1 
            "Qb_dummy" : Qb_dummy, #1,1
            "R_dummy" : R_dummy, #1,1
            "k" : k, #1,1
        }

    def generate_dataset(self, save_path="vehicle_dataset.pkl"):
        """Generates and saves the dataset."""
        dataset = []
        
        for _ in tqdm(range(self.num_samples), desc="Generating Data"):
            sample = self.generate_single_sample()
            '''
            # Print values and their shapes
            true_measurements = sample["true_measurements"]
            print(f"true_measurements are {true_measurements} of shape {np.array(true_measurements).shape}")

            residual_measurements = sample["residual_measurements"]
            print(f"residual_measurements are {residual_measurements} of shape {np.array(residual_measurements).shape}")

            Qa_true = sample["Qa_true"]
            print(f"Qa_true is {Qa_true} of shape {np.array(Qa_true).shape}")

            Qb_true = sample["Qb_true"]
            print(f"Qb_true is {Qb_true} of shape {np.array(Qb_true).shape}")

            R_true = sample["R_true"]
            print(f"R_true is {R_true} of shape {np.array(R_true).shape}")

            plant_yawrate = sample["plant_yawrate"]
            print(f"plant_yawrate is {plant_yawrate} of shape {np.array(plant_yawrate).shape}")

            plant_beta = sample["plant_beta"]
            print(f"plant_beta is {plant_beta} of shape {np.array(plant_beta).shape}")

            plant_Xcg = sample["plant_Xcg"]
            print(f"plant_Xcg is {plant_Xcg} of shape {np.array(plant_Xcg).shape}")

            plant_Ycg = sample["plant_Ycg"]
            print(f"plant_Ycg is {plant_Ycg} of shape {np.array(plant_Ycg).shape}")

            Qa_dummy = sample["Qa_dummy"]
            print(f"Qa_dummy is {Qa_dummy} of shape {np.array(Qa_dummy).shape}")

            Qb_dummy = sample["Qb_dummy"]
            print(f"Qb_dummy is {Qb_dummy} of shape {np.array(Qb_dummy).shape}")

            R_dummy = sample["R_dummy"]
            print(f"R_dummy is {R_dummy} of shape {np.array(R_dummy).shape}")

            k = sample["k"]
            print(f"k is {k} of shape {np.array(k).shape}")
            '''

            dataset.append(sample)

        # Save as pickle file
        with open(save_path, "wb") as f:
            pickle.dump(dataset, f)

        print(f"Dataset saved to {save_path}")

if __name__ == "__main__":
    generator = DataGenerator(num_samples=50000)
    generator.generate_dataset()
