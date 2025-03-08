import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

# Import Kalman filter and vehicle dynamics model
sys.path.insert(0, "/home/asalvi/code_workspace/RL_AdpEst/MachineLearning/VehicleModel/")
from VDkalman_filter_control import KalmanFilterWithControl
from VDlabelQR import LabelQR

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
        self.model = KalmanNet().cuda()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def run(self, output_csv="validation_results.csv"):
        # Storage for logging results
        Qa_hat_values, Qb_hat_values, R_hat_values = [], [], []
        Qa_true_values, Qb_true_values, R_true_values = [], [], []
        beta_hat_all, yaw_hat_all = [], []
        P_updated_all = []

        # Generate first 100 samples
        start = 1
        validate = 1
        LbQR = LabelQR(start, validate)
        values = LbQR.getQ()
        Q_true = values["Q"]  # Label Q
        R_true = np.random.uniform(0, 0.1)  # Label R
        true_measurements = values["TrueMeasureYawRate"] + np.sqrt(Q_true[1, 1]) * np.random.randn(len(values["TrueMeasureYawRate"])) + np.sqrt(R_true) * np.random.randn(len(values["TrueMeasureYawRate"]))

        Qa_true_values = 1500*[Q_true[0, 1]]
        Qb_true_values =  1500*[Q_true[1, 1]]
        R_true_values = 1500*[R_true]

        ### state measurements for baseline
        plant_yawrate = values["TrueYawRate"] 
        plant_beta = values["TrueBeta"] 
        plant_Xcg = values["TrueXcg"] 
        plant_Ycg = values["TrueYcg"]
        ############

        # Initialize rolling window storage
        tmz = np.array(true_measurements[:100])  # Start with first 100 samples
        residual_measurements = np.zeros(100)  # Initial residuals (zero for first steps)
        Qa_hat_values = [np.random.uniform(0, 0.1)] * 100
        Qb_hat_values = [np.random.uniform(0, 0.1)] * 100
        R_hat_values = [np.random.uniform(0, 0.1)] * 100

        # Initialize Kalman Filter
        kf_true = KalmanFilterWithControl(
            x0=np.array([0, 0]), 
            P0=np.array([[1, 0], [0, 1]]), 
            Q=np.diag([Qa_hat_values[-1], Qb_hat_values[-1]]), 
            R=R_hat_values[-1], 
            dt=0.01
        )

        x_hat_states = []
        P_updated = []

        # First 100 steps: only initialize
        for i in range(100):
            kf_true.predict()
            kf_true.update(true_measurements[i])
            residual_measurements[i] = kf_true.y.item()  # Store innovation sequence

            x_hat_states.append(kf_true.x.copy())  # Store estimates
            P_updated.append([np.sqrt(kf_true.P[0, 0]), np.sqrt(kf_true.P[1, 1])])

        # Recursive Estimation Loop
        for k in tqdm(range(101, 1500), desc="Running Prediction"):
            # Generate new measurement sample
            start = k
            validate = 1
            LbQR = LabelQR(start, validate)
            values = LbQR.getQ()
            #Q_true = values["Q"]
            #R_true = np.random.uniform(0, 0.1)
            new_measurement = values["TrueMeasureYawRate"] + np.sqrt(Q_true[1, 1]) * np.random.randn() + np.sqrt(R_true) * np.random.randn()

            #Qa_true_values.append(Q_true[0, 1])
            #Qb_true_values.append(Q_true[1, 1])
            #R_true_values.append(R_true)

            # Append new measurement to rolling window (maintain length 100)
            tmz = np.concatenate([tmz[-99:], [new_measurement[0].item()]])

            # Convert NumPy rolling window to Tensor (ensures 100 elements)
            input_measurements = torch.tensor(tmz, dtype=torch.float32).cuda()
            #print(f"input measurement shape{input_measurements.size()}")
            innovation_seq = torch.tensor(residual_measurements[-100:], dtype=torch.float32).cuda()
            #print(f"innovation_seq shape{innovation_seq.size()}")

            # Previous estimated noise parameters
            dummyQa = torch.tensor(Qa_hat_values[-1], dtype=torch.float32).cuda().unsqueeze(0)
            dummyQb = torch.tensor(Qb_hat_values[-1], dtype=torch.float32).cuda().unsqueeze(0)
            dummyR = torch.tensor(R_hat_values[-1], dtype=torch.float32).cuda().unsqueeze(0)

            # Concatenate inputs and run model prediction
            inputs = torch.cat([input_measurements, innovation_seq, dummyQa, dummyQb, dummyR]).unsqueeze(0)  # Add batch dimension
            
            # Debug: Print the shape to confirm it's always 203
            #print(f"Model input shape: {inputs.shape}")  # Should always be (1, 203)

            predictions = self.model(inputs).cpu().detach().numpy()[0]  # Convert output to NumPy

            # Extract predictions
            Qa_hat, Qb_hat, R_hat = predictions
            Qa_hat_values.append(Qa_hat)
            Qb_hat_values.append(Qb_hat)
            R_hat_values.append(R_hat)

            # Update Kalman filter with new estimated noise parameters
            kf_adaptive = KalmanFilterWithControl(
                x0=np.array(x_hat_states[-1]), 
                P0=np.array([[P_updated[-1][0], 0], [0, P_updated[-1][1]]]), 
                Q=np.array([[Qa_hat, 0], [0, Qb_hat]]), 
                R=R_hat, 
                dt=0.01
            )

            kf_adaptive.predict()
            kf_adaptive.update(tmz[-1])

            # Append latest innovation to residuals (keep last 100)
            residual_measurements = np.concatenate([residual_measurements[-99:], [kf_adaptive.y.item()]])

            # Store Kalman Filter state estimates
            x_hat_states.append(kf_adaptive.x.copy())
            P_updated.append([np.sqrt(kf_adaptive.P[0, 0]), np.sqrt(kf_adaptive.P[1, 1])])

        # Convert lists to NumPy arrays
        x_hat_states = np.array(x_hat_states)
        beta_hat_all = x_hat_states[:, 0]
        yaw_hat_all = x_hat_states[:, 1]

        print(len(Qa_true_values[0:1499]))
        print(len(Qb_true_values[0:1499]))
        print(len(R_true_values[0:1499]))
        print(len(Qa_hat_values[0:1499]))
        print(len(Qb_hat_values[0:1499]))
        print(len(R_hat_values[0:1499]))
        print(len(beta_hat_all[0:1499]))
        print(len(yaw_hat_all[0:1499]))
        print(len(plant_beta[0:1499]))
        print(len(plant_yawrate[0:1499]))
        print(len(plant_Xcg[0:1499]))
        print(len(plant_Ycg[0:1499]))
        print(len([x[0] for x in P_updated][0:1499]))
        print(len([x[1] for x in P_updated][0:1499]))


        # Save Results to CSV
        df = pd.DataFrame({
            "Qa_true": Qa_true_values[0:1499],
            "Qb_true": Qb_true_values[0:1499],
            "R_true": R_true_values[0:1499],
            "Qa_hat": Qa_hat_values[0:1499],
            "Qb_hat": Qb_hat_values[0:1499],
            "R_hat": R_hat_values[0:1499],
            "beta_hat": beta_hat_all[0:1499],
            "yaw_hat": yaw_hat_all[0:1499],
            "beta_true": plant_beta[0:1499],
            "yawRate_true": plant_yawrate[0:1499],
            "Xcg_true": plant_Xcg[0:1499],
            "Ycg_true": plant_Ycg[0:1499],
            "P1_updated": [x[0] for x in P_updated][0:1499],
            "P2_updated": [x[1] for x in P_updated][0:1499],
        })

        df.to_csv(output_csv, index=False)
        print(f"Validation results saved to {output_csv}")

# ------------------------------ RUN VALIDATION -------------------------------- #
if __name__ == "__main__":
    model_path = "/home/asalvi/code_workspace/RL_AdpEst/MachineLearning/VehicleModel/outputC/bslnC_kalman_nn.pth"
    validator = ValidateModel(model_path)
    validator.run(output_csv="/home/asalvi/code_workspace/RL_AdpEst/MachineLearning/VehicleModel/outputC/validation_resultsTEST.csv")
