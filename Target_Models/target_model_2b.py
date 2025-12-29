# Target_Models/target_model_2a_128.py
import torch.nn as nn
import torch
import torch.nn.functional as F

class TargetModel_2b(nn.Module):
    def __init__(self, input_channels=3, output_size=10):
        super(TargetModel_2b, self).__init__()
        # Block 1: 3 -> 16 filters
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # Reduces 128x128 to 64x64
        
        # Block 2: 16 -> 32 filters
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # After second MaxPool: 32x32
        
        # For 128x128 input:
        # After two MaxPool (128/2/2 = 32), image is 32x32
        # So: 32 filters * 32 * 32 = 32768 features
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, output_size)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Output: 16x64x64
        x = self.pool(F.relu(self.conv2(x)))  # Output: 32x32x32
        x = x.view(-1, 32 * 32 * 32)          # Flatten to 32768
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x