import torch.nn as nn
import torch
import torch.nn.functional as F

class TargetModel_2a(nn.Module):
    def __init__(self, input_channels=3, output_size=10):
        super(TargetModel_2a, self).__init__()
        
        # Conv Block 1: 32 filters
        # Input (3, 32, 32) -> (32, 32, 32) -> MaxPool (32, 16, 16)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Conv Block 2: 64 filters
        # Input (32, 16, 16) -> (64, 16, 16) -> MaxPool (64, 8, 8)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # FC Layers: After 2 pools, 32x32 image becomes 8x8
        # 64 filters * 8 * 8 = 4096 units
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        
        # Final Output Layer
        self.output = nn.Linear(128, output_size)

    def forward(self, x):
        # If input is flattened (Batch, 3072), reshape back to (Batch, 3, 32, 32)
        if len(x.shape) == 2:
            x = x.view(-1, 3, 32, 32)
            
        # Conv Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Conv Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Flatten for FC layers: (Batch, 64, 8, 8) -> (Batch, 4096)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Final output (logits)
        x = self.output(x)
        return x