import numpy as np
import pickle
import torch
import sys
from tqdm import tqdm

sys.path.insert(0, "/home/asalvi/code_workspace/RL_AdpEst/MachineLearning/")
from kalman_filter_control import KalmanFilterWithControl

class DataGenerator:
    def __init__(self, num_samples=50000, filter_timesteps=100, n_filters=3):
        self.num_samples = num_samples
        self.filter_timesteps = filter_timesteps
        self.n_filters = n_filters

    def generate_single_sample(self):
        """Generates one data sample with true and simulated measurements."""
        # Random Q_true and R_true
        Q_true = np.random.uniform(0, 2)
        R_true = np.random.uniform(0, 2)

        # Initialize Kalman filter for ground truth measurements
        kf_true = KalmanFilterWithControl(
            x0=np.array([0, 0]), 
            P0=np.array([[1, 0], [0, 1]]), 
            Q=Q_true, 
            R=R_true, 
            dt=0.1
        )

        true_measurements = []
        for _ in range(self.filter_timesteps):
            kf_true.predict()
            measurement = np.random.normal(0, R_true)  # Noisy observation
            kf_true.update(measurement)
            true_measurements.append(measurement)

        sim_measurements = []
        sim_Q = []
        sim_R = []

        # Generate simulated measurements from 3 different Kalman filters
        for _ in range(self.n_filters):
            Q_sim = np.random.uniform(0, 2)
            R_sim = np.random.uniform(0, 2)
            sim_Q.append(Q_sim)
            sim_R.append(R_sim)

            kf_sim = KalmanFilterWithControl(
                x0=np.array([0, 0]), 
                P0=np.array([[1, 0], [0, 1]]), 
                Q=Q_sim, 
                R=R_sim, 
                dt=0.1
            )

            sim_measurements_single = []
            for t in range(self.filter_timesteps):
                kf_sim.predict()
                kf_sim.update(true_measurements[t])  # Use true measurement
                sim_measurements_single.append(kf_sim.predy)
            
            sim_measurements.append(sim_measurements_single)

        return {
            "true_measurements": true_measurements,
            "sim_measurements": sim_measurements,  # List of 3 simulated measurement sequences
            "sim_Q": sim_Q,  # List of 3 Q values
            "sim_R": sim_R,  # List of 3 R values
            "Q_true": Q_true,
            "R_true": R_true,
        }

    def generate_dataset(self, save_path="kalman_dataset.pkl"):
        """Generates and saves the dataset."""
        dataset = []
        
        for _ in tqdm(range(self.num_samples), desc="Generating Data"):
            sample = self.generate_single_sample()
            dataset.append(sample)

        # Save as pickle file
        with open(save_path, "wb") as f:
            pickle.dump(dataset, f)

        print(f"Dataset saved to {save_path}")

if __name__ == "__main__":
    generator = DataGenerator(num_samples=50000)
    generator.generate_dataset()
