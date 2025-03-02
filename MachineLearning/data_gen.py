import numpy as np
import sys
import pandas as pd
sys.path.insert(0, "/home/asalvi/code_workspace/RL_AdpEst/MachineLearning/")

from kalman_filter_control import KalmanFilterWithControl


class KalmanDataGen():
    def __init__(self, Q_true, R_true, n_filters, filter_timesteps):
        self.Q_true = Q_true
        self.R_true = R_true
        self.n_filters = n_filters
        self.filter_timesteps = filter_timesteps
        self.dt = 0.1
        self.x0 = np.array([0, 0])
        self.P0 = np.array([[1, 0], [0, 1]])
        self.H = np.array([1, 0])
        self.true_measurements = []
        self.sim_measurements = np.zeros((filter_timesteps, n_filters))

    def gen_true_measurements(self):
        self.kf = KalmanFilterWithControl(self.x0, self.P0, self.Q_true, self.R_true, self.dt)
        
        for i in range(self.filter_timesteps):
            self.kf.predict()
            measurement = np.dot(self.H, self.kf.gT) + np.random.normal(0, self.R_true)
            self.kf.update(measurement[0])
            self.true_measurements.append(measurement[0])
        
        return self.true_measurements
    
    def gen_sim_measurements(self):
        for i in range(self.n_filters):
            Q = np.random.uniform(0, 2)  # Random initialization
            R = np.random.uniform(0, 2)
            self.kf = KalmanFilterWithControl(self.x0, self.P0, Q, R, self.dt)
            
            for j in range(self.filter_timesteps):
                self.kf.predict()
                measurement = self.true_measurements[j]
                self.kf.update(measurement)
                self.sim_measurements[j, i] = self.kf.predy
        
        return self.sim_measurements
    
    def save_to_csv(self, filename):
        data = {
            'True Measurements': self.true_measurements
        }
        for i in range(self.n_filters):
            data[f'Sim Measurement {i+1}'] = self.sim_measurements[:, i]
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")


if __name__ == "__main__":
    Q_true = 0.1
    R_true = 0.1
    n_filters = 3
    filter_timesteps = 50
    
    generator = KalmanDataGen(Q_true, R_true, n_filters, filter_timesteps)
    generator.gen_true_measurements()
    generator.gen_sim_measurements()
    generator.save_to_csv("kalman_measurements.csv")
