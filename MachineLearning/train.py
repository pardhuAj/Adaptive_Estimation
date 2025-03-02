import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

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
    def __init__(self, input_size=406, hidden_size=128):  # Update input size to 406
        super(KalmanNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)  # Output Q_true, R_true

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Training Function
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
            loss = criterion(predictions, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.6f}")

    torch.save(model.state_dict(), "kalman_nn.pth")
    print("Model saved as kalman_nn.pth")

# Validation Function
def validate_model(dataset_path, model_path, num_samples=100, output_csv="validation_results.csv"):
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
    # Train Model
    train_model("kalman_dataset.pkl", epochs=20)

    # Validate Model
    validate_model("kalman_dataset.pkl", "kalman_nn.pth")