import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import Kalman filter and vehicle dynamics model
import sys
sys.path.insert(0, "/home/asalvi/code_workspace/RL_AdpEst/MachineLearning/VehicleModel/")
from VDkalman_filter_control import KalmanFilterWithControl
from VDlabelQR import LabelQR
from VDdata_gen_dataset import DataGenerator  # ✅ Importing DataGenerator directly

from VDFourwheelmodel_Plots import fourwheel_model
from VDBIcyclemodel_BMW import VehicleModel

# ------------------------------ NEURAL NETWORK MODEL -------------------------------- #
class KalmanNet(nn.Module):
    def __init__(self, input_size=203, hidden_size=128):  # Updated input size
        super(KalmanNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 3)  # Output Qa_hat, Qb_hat, R_hat

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.relu(x)  # Ensure Q_hat and R_hat are non-negative

# ------------------------------ VALIDATION FUNCTION -------------------------------- #
class ValidateModel:
    def __init__(self, model_path):
        #self.num_samples = num_samples
        self.model = KalmanNet().cuda()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        # ✅ Use DataGenerator to create new samples
        self.data_generator = DataGenerator(num_samples=1)

    def run(self, output_csv="validation_results.csv"):
        Qa_hat_values, Qb_hat_values, R_hat_values = [], [], []
        Qa_true_values, Qb_true_values, R_true_values = [], [], []
        beta_true_all, yaw_true_all = [], []
        Xcg_true_all, Ycg_true_all = [], []  # ✅ New tracking
        beta_hat_all, yaw_hat_all = [], []
        P_updated_all = []

        #print(f"Starting validation with {self.num_samples} generated samples...")

        #for _ in tqdm(range(self.num_samples), desc="Running Validation"):

        # ✅ Generate new sample using DataGenerator
        sample = self.data_generator.generate_single_sample()
        #print(sample)

        # Convert NumPy lists to NumPy arrays before converting to tensor
        true_measurements = torch.tensor(np.array(sample["true_measurements"][:100]), dtype=torch.float32).squeeze()
        #print(true_measurements.shape)
        residual_measurements = torch.tensor(np.array(sample["residual_measurements"]), dtype=torch.float32).squeeze()
        #print(residual_measurements.shape)

        dummyQa = torch.tensor(sample["Qa_dummy"], dtype=torch.float32).unsqueeze(0)
        dummyQb = torch.tensor(sample["Qb_dummy"], dtype=torch.float32).unsqueeze(0)
        dummyR = torch.tensor(sample["R_dummy"], dtype=torch.float32).unsqueeze(0)


        # Concatenate inputs
        inputs = torch.cat([true_measurements, residual_measurements, dummyQa, dummyQb, dummyR]).cuda()


        predictions = self.model(inputs)  # Shape: [batch_size, 3]

        # Extract True Measurements from Inputs (Shape: [batch_size, 100])
        true_measurementsZ = np.array(sample["true_measurements"][:100])

        # Extract Qa_hat, Qb_hat, R_hat for Each Batch (Shape: [batch_size])
        predictions = predictions.cpu().detach().numpy()  # Convert to NumPy
        Qa_hat, Qb_hat, R_hat = predictions[0], predictions[1], predictions[2]

        # **Adaptive Kalman Filter Tracking Only 2 States: Beta & Yaw**
        # Initialize Kalman filter for ground truth measurements
        kf_adaptive = KalmanFilterWithControl(
            x0=np.array([0, 0]), 
            P0=np.array([[1, 0], [0, 1]]), 
            Q = [[Qa_hat.item(),0],[0,Qb_hat.item()]], 
            R = R_hat.item(), 
            dt=0.01
            )

        x_hat_states = []
        P_updated = []

        

        for _ in range(100):  # Process all 100 timesteps
            kf_adaptive.predict()
            #print(f"size of tm{np.shape(true_measurements)}")
            kf_adaptive.update(true_measurementsZ[_])

            x_hat_states.append(kf_adaptive.x.copy())  # Store adaptive estimates

            p_1 = np.sqrt(kf_adaptive.P[0, 0])  # Uncertainty in beta
            p_2 = np.sqrt(kf_adaptive.P[1, 1])  # Uncertainty in yaw
            P_updated.append([p_1, p_2])

        
        
        modelBicyle = VehicleModel(25)
        BicyleManeuver = modelBicyle.run_simulation(1) 
        Bi_states = np.array(BicyleManeuver["x"])

        Bi_beta = Bi_states[0,:100] + np.random.normal(0,0.01*np.sqrt(sample["Qa_true"]))
        print(f"Bi_beta shape is : {np.shape(Bi_beta)}")
        Bi_yaw = Bi_states[1,:100] + np.random.normal(0,0.01*np.sqrt(sample["Qb_true"]))

        # Store logged data
        #Qa_hat_values.append(Qa_hat)
        #Qb_hat_values.append(Qb_hat)
        #R_hat_values.append(R_hat)
        #Qa_true_values.append(sample["Qa_true"])
        #Qb_true_values.append(sample["Qb_true"])
        #R_true_values.append(sample["R_true"])

        # ✅ Log final values for comparison
        #beta_true_all.append(sample["plant_beta"])
        #yaw_true_all.append(sample["plant_yawrate"])
        #Xcg_true_all.append(sample["plant_Xcg"])
        #Ycg_true_all.append(sample["plant_Ycg"])
            
        print(np.shape(x_hat_states))
        x_hat_states = np.array(x_hat_states)
        beta_hat_all = x_hat_states[:,0]
        yaw_hat_all = x_hat_states[:,1]
        P_updated_all = P_updated
        print(np.shape(sample["plant_beta"][:100]))
        print(np.shape(np.ones(100)*sample["Qa_true"]))
        print(np.shape(beta_hat_all))

    #print(f"x_true shape: {len(yaw_true_all)}")
    #print(f"x_hat shape: {len(yaw_hat_all)}")

        # Convert data to DataFrame
        df = pd.DataFrame({
            "Qa_true": np.ones(100)*sample["Qa_true"],
            "Qb_true": np.ones(100)*sample["Qb_true"],
            "R_true": np.ones(100)*sample["R_true"],
            "Qa_hat": np.ones(100)*Qa_hat,
            "Qb_hat": np.ones(100)*Qb_hat,
            "R_hat": np.ones(100)*R_hat,
            "beta_true": sample["plant_beta"][:100],
            "yaw_true": sample["plant_yawrate"][:100],
            "Xcg_true": sample["plant_Xcg"][:100],
            "Ycg_true": sample["plant_Ycg"][:100],
            "beta_hat": beta_hat_all,
            "yaw_hat": yaw_hat_all,
            "P1_updated": [x[0] for x in P_updated_all],
            "P2_updated": [x[1] for x in P_updated_all],
            "biBeta" : Bi_beta,
            "biYaw" : Bi_yaw
        })

        df.to_csv(output_csv, index=False)
        print(f"Validation results saved to {output_csv}")

# ------------------------------ RUN VALIDATION -------------------------------- #
if __name__ == "__main__":
    model_path = "/home/asalvi/code_workspace/RL_AdpEst/MachineLearning/VehicleModel/outputB/PINNB_kalman_nn.pth"
    validator = ValidateModel(model_path)
    validator.run(output_csv="/home/asalvi/code_workspace/RL_AdpEst/MachineLearning/VehicleModel/outputB/PINNB_validation_results.csv")
