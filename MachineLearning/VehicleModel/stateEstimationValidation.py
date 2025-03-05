import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import your Kalman filter and utility functions
import sys
sys.path.insert(0, "/home/asalvi/code_workspace/RL_AdpEst/MachineLearning/")
from kalman_filter_control import KalmanFilterWithControl

# ------------------------------ NEURAL NETWORK MODEL -------------------------------- #
class KalmanNet(nn.Module):
    """Neural network to predict Q and R"""
    def __init__(self, input_size=406, hidden_size=128):
        super(KalmanNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)  # Output Q_hat, R_hat

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.relu(x)  # Ensure Q_hat and R_hat are non-negative

# ------------------------------ DATA COLLECTION CLASS -------------------------------- #
class CollectData:
    def __init__(self, filter_timesteps=100, n_filters=3):
        self.filter_timesteps = filter_timesteps
        self.n_filters = n_filters

    def generate_single_sample(self, Q_true, R_true, Q_sim, R_sim):
        """Generates one data sample with true and simulated measurements."""
        kf_true = KalmanFilterWithControl(
            x0=np.array([0, 0]), 
            P0=np.array([[1, 0], [0, 1]]), 
            Q=Q_true, 
            R=R_true, 
            dt=0.1
        )

        true_measurements, true_states = [], []
        for _ in range(self.filter_timesteps):
            kf_true.predict()
            measurement = np.random.normal(0, R_true)
            kf_true.update(measurement)
            true_measurements.append(measurement)
            true_states.append(kf_true.x.copy())  # Store true state estimate

        # Generate Simulated Measurements
        sim_measurements = []
        for i in range(self.n_filters):
            kf_sim = KalmanFilterWithControl(
                x0=np.array([0, 0]), 
                P0=np.array([[1, 0], [0, 1]]), 
                Q=Q_sim[i], 
                R=R_sim[i], 
                dt=0.1
            )

            sim_measurements_single = []
            for meas in true_measurements:
                kf_sim.predict()
                kf_sim.update(meas)  # Use true measurement
                sim_measurements_single.append(kf_sim.predy)

            sim_measurements.append(sim_measurements_single)

        return {
            "true_measurements": true_measurements,
            "sim_measurements": sim_measurements,  # Added simulated measurements
            "Q_true": Q_true,
            "R_true": R_true,
            "true_states": np.array(true_states)  # Store for logging
        }


# ------------------------------ SIMULATION LOOP -------------------------------- #
class Simulate:
    def __init__(self, model_path, timesteps=1000, window_size=100):
        self.timesteps = timesteps
        self.window_size = window_size
        self.model = KalmanNet().cuda()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def run(self):
        collect_data = CollectData()
        Q_true, R_true = np.random.uniform(0, 2), np.random.uniform(0, 2)

        Q_hat_values, R_hat_values = [], []
        Q_true_values, R_true_values = [], []
        x_true_all, x_hat_all, P_updated_all= [], [], []

        for t in tqdm(range(self.timesteps), desc="Running Simulation"):
            # Change Q_true and R_true every 250 timesteps
            if t % 250 == 0:
                Q_true, R_true = np.random.uniform(0, 2), np.random.uniform(0, 2)
                print(f"Timestep {t}: Q_true={Q_true}, R_true={R_true}")

            Q_sim = np.random.uniform(0, 2, 3)
            R_sim = np.random.uniform(0, 2, 3)

            # Collect Data
            sample = collect_data.generate_single_sample(Q_true, R_true, Q_sim, R_sim)

            true_measurements = np.array(sample["true_measurements"])
            true_states = sample["true_states"]  # Shape (100, 2)

            # Prepare input for NN
            inputs = torch.tensor(np.concatenate([true_measurements, Q_sim, R_sim]), dtype=torch.float32).cuda().unsqueeze(0)
            sim_measurements = np.array(sample["sim_measurements"]).flatten()  # Flatten 3x100 -> 300
            inputs = torch.tensor(np.concatenate([true_measurements, sim_measurements, Q_sim, R_sim]), dtype=torch.float32).cuda().unsqueeze(0)


            # Predict Q_hat and R_hat using NN
            with torch.no_grad():
                Q_hat, R_hat = self.model(inputs).cpu().numpy().flatten()

            # Run Kalman filter with predicted Q_hat and R_hat
            kf_adaptive = KalmanFilterWithControl(
                x0=np.array([0, 0]), 
                P0=np.array([[1, 0], [0, 1]]), 
                Q=Q_hat, 
                R=R_hat, 
                dt=0.1
            )

            x_hat_states = []
            P_updated = []
            for meas in true_measurements:
                kf_adaptive.predict()
                kf_adaptive.update(meas)
                x_hat_states.append(kf_adaptive.x.copy())  # Store adaptive estimates
                #print(kf_adaptive.P[0,0])
                p_1 = np.sqrt(kf_adaptive.P[0,0])
                #print(p_1)
                p_2 = np.sqrt(kf_adaptive.P[1,1])
                P = [p_1,p_2]
                P_updated.append(P)

            # Store logged data
            Q_hat_values.append(Q_hat)
            R_hat_values.append(R_hat)
            Q_true_values.append(Q_true)
            R_true_values.append(R_true)
            x_true_all.append(true_states[-1])  # Final state estimate from true Q/R
            x_hat_all.append(x_hat_states[-1])  # Final state estimate from predicted Q_hat/R_hat
            P_updated_all.append(P_updated[-1])

        # Convert data to DataFrame
        df = pd.DataFrame({
            "Q_true": Q_true_values,
            "R_true": R_true_values,
            "Q_hat": Q_hat_values,
            "R_hat": R_hat_values,
            "x_true_1": [x[0] for x in x_true_all],
            "x_true_2": [x[1] for x in x_true_all],
            "x_hat_1": [x[0] for x in x_hat_all],
            "x_hat_2": [x[1] for x in x_hat_all],
            "P1_updated": [x[0] for x in P_updated_all],
            "P2_updated": [x[1] for x in P_updated_all],
        })

        df.to_csv("simulation_results.csv", index=False)
        print("Simulation results saved to simulation_results.csv")

# ------------------------------ RUN SIMULATION -------------------------------- #
if __name__ == "__main__":
    model_path = "PINN_kalman_nn.pth"
    simulator = Simulate(model_path, timesteps=1000)
    simulator.run()
