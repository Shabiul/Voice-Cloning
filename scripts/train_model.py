import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class AudioDataset(Dataset):
    def __init__(self, data_dir):
        """
        Initialize the dataset.
        
        Args:
        - data_dir (str): Path to the directory containing .pt files.
        """
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_path = os.path.join(self.data_dir, file_name)
        tensor = torch.load(file_path)
        return tensor, tensor  # Assuming the task is self-supervised or reconstruction

import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # Define a placeholder for the input size
        self._to_linear = None
        self._initialize_linear_layers()

        self.fc1 = nn.Linear(self._to_linear, 128)  # Update input features based on calculation
        self.fc2 = nn.Linear(128, 10)  # Assuming 10 output classes, adjust if needed
    
    def _initialize_linear_layers(self):
        # Create a dummy input tensor to determine the output size of conv layers
        x = torch.zeros(1, 1, 64, 64)  # Example input size, adjust as needed
        x = self.conv1(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(1, -1)  # Flatten the tensor
        self._to_linear = x.size(1)  # Get the size of the flattened tensor

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(data_dir, epochs=10, batch_size=16, learning_rate=0.001):
    """
    Train the model.
    
    Args:
    - data_dir (str): Directory containing the .pt files.
    - epochs (int): Number of training epochs.
    - batch_size (int): Batch size.
    - learning_rate (float): Learning rate for the optimizer.
    """
    # Initialize dataset and dataloader
    dataset = AudioDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model, loss function, and optimizer
    model = SimpleCNN()
    criterion = nn.MSELoss()  # Use appropriate loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# Set the directory for processed audio
processed_audio_dir = "data/processed_audio"

# Train the model
train_model(processed_audio_dir)
