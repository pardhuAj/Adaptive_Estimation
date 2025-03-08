
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

#from VDFourwheelmodel_Plots import fourwheel_model
from VDBIcyclemodel_BMW import VehicleModel

class LabelQR:
    def __init__(self,k,validate):
        #self.randM = MassRand
        #self.randIz = InertiaRand
        #self.TMYaw  = None
        self.k = k
        self.validate = validate


    def compare(self):

        velocity = 25 #25 m/s
        #maneuver_type = 1 #1:Fishook 2:skidpad 3:Slalom
        # Simulate Plant model with randomization
        
        #print(f"Rand M is:{randM}")
        #print(f"Rand Iz is:{randIz}")
        validation = {"enable" : self.validate, "cf":np.random.uniform(0,0.15)*(2*802 * 180 / np.pi), "cr":np.random.uniform(0,0.15)*(2*802 * 180 / np.pi), "validation_cf": 0.5, "validation_cr": 0.5 }
        modelNominal = VehicleModel(velocity, validation)
        maneuver_type = np.random.randint(low = 1,high=4)
        NmManeuver = modelNominal.run_simulation(maneuver_type)
        
        NM_states = np.array(NmManeuver["x"])
        #print(FW_states.shape)

        NM_beta = NM_states[0,:]
        #print(FW_beta)
        NM_yawRate = NM_states[1,:]
        NM_Xcg = NM_states[2,:]
        NM_Ycg = NM_states[3,:]
        self.Truebeta = NM_beta
        self.TrueYawRate = NM_yawRate
        self.TrueXcg = NM_Xcg
        self.TrueYcg = NM_Ycg


        

        #Beta 
        validation = {"enable" : self.validate, "cf":0, "cr":0, "validation_cf": 0.5, "validation_cr": 0.5}
        modelBicyle = VehicleModel(velocity, validation)
        BicyleManeuver = modelBicyle.run_simulation(maneuver_type) 

        Bi_states = np.array(BicyleManeuver["x"])

        Bi_beta = Bi_states[0,:]
        Bi_yawRate = Bi_states[1,:]
        Bi_Xcg = Bi_states[2,:]
        Bi_Ycg = Bi_states[3,:]

        #k = np.random.randint(low = 1, high = 1500)
        #self.k = k
        self.TrueMeasureYawRate = Bi_yawRate[self.k:self.k+100] #A non-varyng model to which Q and R will be added 

        #error_beta = NM_beta[self.k:self.k+100] - Bi_beta[self.k:self.k+100]
        #error_yaw = NM_yawRate[self.k:self.k+100] - Bi_yawRate[self.k:self.k+100]

        error_beta = NM_beta- Bi_beta
        error_yaw = NM_yawRate - Bi_yawRate


        return {
            "error_beta" : error_beta,
            "error_yawRate" : error_yaw,
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
        Q_yaw_var = np.var(errors["error_yawRate"]) + eps

        # Return Q as a diagonal covariance matrix
        Q = np.diag([Q_beta_var, Q_yaw_var])
        #print(Q)
        
        return {
            "Q" : Q,
            "TrueBeta" : self.Truebeta,
            "TrueYawRate" : self.TrueYawRate,
            "TrueXcg" : self.TrueXcg,
            "TrueYcg" : self.TrueYcg,
            "TrueMeasureYawRate" : self.TrueMeasureYawRate, #these are randomly drawn 100 samples for training dataset
            "k" : self.k, #these are where the samples begin
        }




if __name__== "__main__":

    '''
    
    for _ in range(1):
        LbQR = LabelQR()
        values = LbQR.getQ()
        Q = values["Q"]
        print(f" Q is : {Q} of the size {Q.shape}")
        TrueBeta = values["TrueBeta"]
        print(f"TrueBeta is : {TrueBeta} of the size {TrueBeta.shape}")
        TrueYawRate = values["TrueYawRate"]
        print(f"TrueYawRate is : {TrueYawRate} of the size {TrueYawRate.shape}")
        TrueXcg = values["TrueXcg"]
        print(f"TrueXcg is : {TrueXcg} of the size {TrueXcg.shape}")
        TrueYcg = values["TrueYcg"]
        print(f"TrueYcg is : {TrueYcg} of the size {TrueYcg.shape}")
        TrueMeasureYawRate = values["TrueMeasureYawRate"]
        print(f"TrueMeasureYawRate is {TrueMeasureYawRate} of the size {TrueMeasureYawRate.shape}")
        k = values["k"]
        print(f"k is : {k}")

    '''


    #import matplotlib.pyplot as plt
    #plt.figure()
    #plt.plot(values["TrueMeasureBeta"])
    #plt.show()
