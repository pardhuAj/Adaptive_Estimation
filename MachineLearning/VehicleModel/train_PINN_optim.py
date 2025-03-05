import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time

import sys
sys.path.insert(0, "/home/asalvi/code_workspace/RL_AdpEst/MachineLearning/VehicleModel/")
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

        # Convert lists of NumPy arrays into a single NumPy array before creating PyTorch tensor
        true_measurements = torch.tensor(np.array(sample["true_measurements"]), dtype=torch.float32).squeeze()
        residual_measurements = torch.tensor(np.array(sample["residual_measurements"]), dtype=torch.float32).squeeze()

        inputs = torch.cat([true_measurements, residual_measurements])

        target = torch.tensor(
            np.array([sample["Qa_true"], sample["Qb_true"], sample["R_true"]]), dtype=torch.float32
        )

        return inputs, target


class KalmanNet(nn.Module):
    def __init__(self, input_size=200, hidden_size=128):
        super(KalmanNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 3)  # Outputs Qa_true, Qb_true, R_true

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.relu(x)  # Ensure Q_hat and R_hat are non-negative

def train_model(dataset_path, epochs=10, batch_size=128, lr=0.001):
    print(f"Starting Training for {epochs} epochs...\n")
    
    dataset = KalmanDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = KalmanNet().cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0

        for batch_idx, (inputs, target) in enumerate(dataloader):
            inputs, target = inputs.cuda(non_blocking=True), target.cuda(non_blocking=True)
            optimizer.zero_grad()
            
            # Model Prediction
            predictions = model(inputs)  # Shape: [batch_size, 3]
            mse_loss = criterion(predictions, target)

            # **Optimized Kalman Filter Processing** (Vectorized)
            Qa_hat, Qb_hat, R_hat = predictions[:, 0], predictions[:, 1], predictions[:, 2]
            
            # Avoid repeated .cpu().numpy() calls, work directly in PyTorch
            true_measurements = inputs[:, :100]  # Shape: [batch_size, 100]
            residual_measurements_batch = torch.zeros_like(true_measurements)

            for i in range(100):  # Process all time steps together
                kf_pred = KalmanFilterWithControl(
                    torch.zeros(batch_size, 2, device=inputs.device),
                    torch.eye(2, device=inputs.device).expand(batch_size, -1, -1),
                    torch.stack([Qa_hat, Qb_hat], dim=-1).unsqueeze(-1).expand(-1, 2, 2),  
                    R_hat.unsqueeze(-1),
                    0.01
                )
                kf_pred.predict()
                kf_pred.update(true_measurements[:, i])
                residual_measurements_batch[:, i] = kf_pred.y

            # **Parallelized Sample Autocorrelation and NIS Loss Computation**
            J_ = SampleAutocorrelation(true_measurements, residual_measurements_batch).get_J()
            J_ = torch.tensor(J_, dtype=torch.float32, device=inputs.device)

            NIS_ = NIS(true_measurements, residual_measurements_batch).get_NISLoss()
            NIS_ = torch.tensor(NIS_, dtype=torch.float32, device=inputs.device)

            # Compute Total Loss
            total_loss_value = mse_loss + 0.1 * J_ + 0.1 * NIS_
            total_loss_value.backward()
            optimizer.step()
            
            total_loss += total_loss_value.item()

            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)} - Loss: {total_loss_value.item():.6f}")

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {total_loss / len(dataloader):.6f} - Time: {epoch_time:.2f}s\n")

    torch.save(model.state_dict(), "bsln_kalman_nn.pth")
    print("Model saved as bsln_kalman_nn.pth")

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

            predictions = predictions.cpu().numpy()
            targets = targets.cpu().numpy()

            df = pd.DataFrame({
                "Qa_true": targets[:, 0],
                "Qb_true": targets[:, 1],
                "R_true": targets[:, 2],
                "Qa_pred": predictions[:, 0],
                "Qb_pred": predictions[:, 1],
                "R_pred": predictions[:, 2]
            })
            df.to_csv(output_csv, index=False)
            print(f"Validation results saved to {output_csv}")
            break

if __name__ == "__main__":
    train_model("vehicle_dataset.pkl", epochs=20)
    validate_model("vehicle_dataset.pkl", "bsln_kalman_nn.pth", "bsln_validation.csv")
