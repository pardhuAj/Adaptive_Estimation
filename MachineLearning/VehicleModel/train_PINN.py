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

        # Convert NumPy lists to NumPy arrays before converting to tensor
        true_measurements = torch.tensor(np.array(sample["true_measurements"]), dtype=torch.float32).squeeze()
        #print(true_measurements.shape)
        residual_measurements = torch.tensor(np.array(sample["residual_measurements"]), dtype=torch.float32).squeeze()
        #print(residual_measurements.shape)

        # Concatenate inputs
        inputs = torch.cat([true_measurements, residual_measurements])

        # Targets (Q_true, R_true)
        target = torch.tensor([sample["Qa_true"],sample["Qb_true"], sample["R_true"]], dtype=torch.float32)

        return inputs, target

class KalmanNet(nn.Module):
    def __init__(self, input_size=200, hidden_size=128):
        super(KalmanNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 3)  # Output Qa_true, Qb_true , R_true

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.relu(x)  # Ensure Q_hat and R_hat are non-negative


# Training Function with Physics-Informed Loss
def train_model(dataset_path, epochs=10, batch_size=128, lr=0.001):

    print(f"Starting Training for {epochs} epochs...\n")

    dataset = KalmanDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = KalmanNet().cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        start_time = time.time()  # Track epoch start time
        total_loss = 0
        for inputs, target in dataloader:
            inputs, target = inputs.cuda(), target.cuda()
            optimizer.zero_grad()
            
            # Model Prediction
            predictions = model(inputs)  # Shape: [batch_size, 3]
            mse_loss = criterion(predictions, target)

            # Extract True Measurements from Inputs (Shape: [batch_size, 100])
            true_measurements = inputs[:, :100].cpu().numpy()

            # Extract Qa_hat, Qb_hat, R_hat for Each Batch (Shape: [batch_size])
            predictions = predictions.cpu().detach().numpy()  # Convert to NumPy
            Qa_hat, Qb_hat, R_hat = predictions[:, 0], predictions[:, 1], predictions[:, 2]

            residual_measurements_batch = []

            # Loop over each sample in batch
            for batch_idx in range(inputs.shape[0]):  # batch_size iterations
                batch_start = time.time()  # Track batch time
                kf_pred = KalmanFilterWithControl(
                    np.array([0, 0]), 
                    np.array([[1, 0], [0, 1]]), 
                    np.array([[Qa_hat[batch_idx], 0], [0, Qb_hat[batch_idx]]]), 
                    R_hat[batch_idx], 
                    0.01
                )
                
                residual_measurements = []
                for i in range(100):  # Process 100 timesteps
                    kf_pred.predict()
                    kf_pred.update(true_measurements[batch_idx, i])
                    residual_measurements.append(kf_pred.y)

                residual_measurements_batch.append(residual_measurements)  # Store per batch
            
            # Convert residuals into NumPy array for processing
            residual_measurements_batch = np.array(residual_measurements_batch)  # Shape: [batch_size, 100]

            # Compute J_ and NIS Loss per batch
            J_ = SampleAutocorrelation(true_measurements, residual_measurements_batch).get_J()
            J_ = torch.tensor(J_, dtype=torch.float32, device=inputs.device)

            NIS_ = NIS(true_measurements, residual_measurements_batch.squeeze()).get_NISLoss()
            NIS_ = torch.tensor(NIS_, dtype=torch.float32, device=inputs.device)

            # Compute Total Loss
            total_loss_value = mse_loss + 0.1 * J_ + 0.1 * NIS_
            total_loss_value.backward()
            optimizer.step()
            
            total_loss += total_loss_value.item()

            # Print intermediate progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)} - Loss: {total_loss_value.item():.6f} - Time: {time.time() - batch_start:.2f}s")


        epoch_time = time.time() - start_time  # Calculate time per epoch
        print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {total_loss / len(dataloader):.6f} - Time: {epoch_time:.2f}s\n")

    torch.save(model.state_dict(), "bsln_kalman_nn.pth")
    print("Model saved as bsln_kalman_nn.pth")

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
                "Qa_true": targets[:, 0],
                "Qb_true": targets[:, 1],
                "R_true": targets[:, 2],
                "Qa_pred": predictions[:, 0],
                "Qb_pred": predictions[:, 1],
                "R_pred": predictions[:, 2]
            })

            # Save to CSV
            df.to_csv(output_csv, index=False)
            print(f"Validation results saved to {output_csv}")
            break  # Only process the first batch

if __name__ == "__main__":
    train_model("vehicle_dataset.pkl", epochs=20)
    validate_model("vehicle_dataset.pkl", "bsln_kalman_nn.pth","bsln_validation.csv")
