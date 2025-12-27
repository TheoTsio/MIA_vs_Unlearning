import torch.nn as nn
import torch

class TargetModel_1a(nn.Module):
    def __init__(self, input_size=49152, output_size=10):
        super(TargetModel_1a, self).__init__()
        # Layer 1: Input (49152) -> 512
        self.fc1 = nn.Linear(input_size, 512)  
        # Layer 2: 512 -> 256
        self.fc2 = nn.Linear(512, 256)         
        # Layer 3: 256 -> 128
        self.fc3 = nn.Linear(256, 128)         
        # Final Output Layer: 128 -> 10
        self.output = nn.Linear(128, output_size)

    def forward(self, x):
        # Flatten the image: (Batch, 3, 128, 128) -> (Batch, 49152)
        x = x.view(x.size(0), -1) 
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        
        # Final output (logits)
        x = self.output(x)
        return x