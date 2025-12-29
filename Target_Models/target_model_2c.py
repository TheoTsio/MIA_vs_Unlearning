import torch.nn as nn
import torch

class TargetModel_2c(nn.Module):
    def __init__(self, input_size=49152, output_size=10):
        super(TargetModel_2c, self).__init__()
        # Layer 1: Input (49152) -> 256
        self.fc1 = nn.Linear(input_size, 256)
        # Layer 2: 256 -> 128
        self.fc2 = nn.Linear(256, 128)
        # Final Output Layer: 128 -> output_size
        self.output = nn.Linear(128, output_size)
    
    def forward(self, x):
        # Flatten the image: (Batch, 3, 128, 128) -> (Batch, 49152)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # Final output (logits)
        x = self.output(x)
        return x