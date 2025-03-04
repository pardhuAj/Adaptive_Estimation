import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, "/home/asalvi/code_workspace/RL_AdpEst/MachineLearning/")
from kalman_filter_control import KalmanFilterWithControl
from sample_autocorrelation import SampleAutocorrelation 
from nis_ import NIS

class KalmanDataset(Dataset):
    def __init__(self, dataset_path):
        with open(dataset_path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Convert NumPy lists to NumPy arrays before converting to tensor
        true_measurements = torch.tensor(np.array(sample["true_measurements"]), dtype=torch.float32)
        sim_measurements = torch.tensor(np.array(sample["sim_measurements"]), dtype=torch.float32).flatten()  # Flatten 3 x 100 -> 300
        sim_Q = torch.tensor(np.array(sample["sim_Q"]), dtype=torch.float32)
        sim_R = torch.tensor(np.array(sample["sim_R"]), dtype=torch.float32)

        # Concatenate inputs
        inputs = torch.cat([true_measurements, sim_measurements, sim_Q, sim_R])

        # Targets (Q_true, R_true)
        target = torch.tensor([sample["Q_true"], sample["R_true"]], dtype=torch.float32)

        return inputs, target

class KalmanNet(nn.Module):
    def __init__(self, input_size=406, hidden_size=128):
        super(KalmanNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)  # Output Q_true, R_true

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.relu(x)  # Ensure Q_hat and R_hat are non-negative


# Training Function with Physics-Informed Loss
def train_model(dataset_path, epochs=10, batch_size=128, lr=0.001):
    dataset = KalmanDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = KalmanNet().cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for inputs, target in dataloader:
            inputs, target = inputs.cuda(), target.cuda()
            optimizer.zero_grad()
            predictions = model(inputs)
            mse_loss = criterion(predictions, target)

            # Generate predicted measurements using Q_hat and R_hat
            true_measurements = inputs[:, :100].cpu().numpy()
            pred_measurements = []
            for i in range(len(true_measurements)):
                Q_hat, R_hat = predictions[i].cpu().detach().numpy()
                kf_pred = KalmanFilterWithControl(np.array([0, 0]), np.array([[1, 0], [0, 1]]), Q_hat, R_hat, 0.1)
                pred_seq = []
                for t in range(100):
                    kf_pred.predict()
                    measurement = np.dot(np.array([1, 0]), kf_pred.gT) + np.random.normal(0, R_hat)
                    kf_pred.update(measurement[0])
                    pred_seq.append(measurement[0])
                pred_measurements.append(pred_seq)
            pred_measurements = np.array(pred_measurements)
            
            #J_ = np.mean([SampleAutocorrelation(true_measurements[i], pred_measurements[i]).get_J() for i in range(len(true_measurements))])
            J_ = SampleAutocorrelation(true_measurements, pred_measurements).get_J()
            J_ = torch.tensor(J_, dtype=torch.float32, device=inputs.device)

            # Calculate NIS Loss
            NIS_ = NIS(true_measurements, pred_measurements).get_NISLoss()
            NIS_ = torch.tensor(NIS_, dtype=torch.float32, device=inputs.device)
            
            total_loss_value = mse_loss + 0.1 * J_+ 0.1*NIS_
            total_loss_value.backward()
            optimizer.step()
            total_loss += total_loss_value.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.6f}")

    torch.save(model.state_dict(), "PINN_kalman_nn.pth")
    print("Model saved as PINN_kalman_nn.pth")

# Validation Function
def validate_model(dataset_path, model_path, output_csv, num_samples=100):
    dataset = KalmanDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=True)

    model = KalmanNet().cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.cuda(), targets.cuda()
            predictions = model(inputs)

            # Move to CPU for saving
            predictions = predictions.cpu().numpy()
            targets = targets.cpu().numpy()

            # Convert to DataFrame
            df = pd.DataFrame({
                "Q_true": targets[:, 0],
                "R_true": targets[:, 1],
                "Q_pred": predictions[:, 0],
                "R_pred": predictions[:, 1]
            })

            # Save to CSV
            df.to_csv(output_csv, index=False)
            print(f"Validation results saved to {output_csv}")
            break  # Only process the first batch

if __name__ == "__main__":
    #train_model("kalman_dataset.pkl", epochs=20)
    validate_model("kalman_dataset_validation.pkl", "bsln_kalman_nn.pth","bsln_validation.csv")
