
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

class LabelQR:
    def __init__(self,MassRand,InertiaRand):
        self.randM = MassRand
        self.randIz = InertiaRand
        self.TMYaw  = None


    def compare(self):

        velocity = 25 #25 m/s
        maneuver_type = 1 #1:Fishook 2:skidpad 3:Slalom
        # Simulate Plant model with randomization
        
        #print(f"Rand M is:{randM}")
        #print(f"Rand Iz is:{randIz}")
        modelFW = fourwheel_model(velocity,self.randM,self.randIz)
        FWManeuver = modelFW.run_simulation(maneuver_type)
        
        FW_states = np.array(FWManeuver["x"])
        #print(FW_states.shape)

        FW_beta = FW_states[1,:]
        #print(FW_beta)
        FW_yaw = FW_states[0,:]
        FW_Xcg = FW_states[5,:]
        FW_Ycg = FW_states[6,:]
        self.TMbeta = FW_beta
        self.TMYaw = FW_yaw
        self.TMXcg = FW_Xcg
        self.TMYcg = FW_Ycg

        #Beta 

        modelBicyle = VehicleModel(velocity)
        BicyleManeuver = modelBicyle.run_simulation(maneuver_type) 

        Bi_states = np.array(BicyleManeuver["x"])

        Bi_beta = Bi_states[0,:]
        Bi_yaw = Bi_states[1,:]
        Bi_Xcg = Bi_states[2,:]
        Bi_Ycg = Bi_states[3,:]

        error_beta = FW_beta - Bi_beta
        error_yaw = FW_yaw - Bi_yaw


        return {
            "error_beta" : error_beta,
            "error_yaw" : error_yaw,
        }

        '''

        plt.figure()
        plt.plot(FW_beta, label="FW_beta", linestyle="dashed", color="blue", alpha=0.7)
        plt.plot(Bi_beta, label="Bi_beta", linestyle="dashed", color="red", alpha=0.7)
        plt.legend()
        plt.title("Beta")

        plt.figure()
        plt.plot(FW_yaw, label="FW_yaw", linestyle="dashed", color="blue", alpha=0.7)
        plt.plot(Bi_yaw, label="Bi_yaw", linestyle="dashed", color="red", alpha=0.7)
        plt.legend()
        plt.title("Yaw Rate")

        plt.figure()
        plt.plot(FW_Xcg,FW_Ycg, label="FW_pose", linestyle="dashed", color="blue", alpha=0.7)
        plt.plot(Bi_Xcg,Bi_Ycg, label="Bi_pose", linestyle="dashed", color="red", alpha=0.7)
        plt.legend()
        plt.title("Pose")

        plt.show()
        '''

    def getQ(self):
        errors = self.compare()

        # Compute Gaussian parameters (mean and variance)
        eps = 1e-6  # Small value for numerical stability
        Q_beta_var = np.var(errors["error_beta"]) + eps
        Q_yaw_var = np.var(errors["error_yaw"]) + eps

        # Return Q as a diagonal covariance matrix
        Q = np.diag([Q_beta_var, Q_yaw_var])
        #print(Q)
        
        return {
            "Q" : Q,
            "TrueMeasureBeta" : self.TMbeta,
            "TrueMeasureYaw" : self.TMYaw,
            "TrueMeasureXcg" : self.TMXcg,
            "TrueMeasureYcg" : self.TMYcg,
        }




if __name__== "__main__":
    mass_random = np.random.uniform(0,500)
    inertia_random = np.random.uniform(0,100)
    LbQR = LabelQR(mass_random,inertia_random)
    values = LbQR.getQ()

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(values["TrueMeasureBeta"])
    plt.show()
