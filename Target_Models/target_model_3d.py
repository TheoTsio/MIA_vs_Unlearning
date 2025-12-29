import torch.nn as nn
import torch

class TargetModel_3d(nn.Module):
    def __init__(self, input_size=600, output_size=100):
        super(TargetModel_3d, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        # Ensure input is flat (batch_size, 600)
        return self.layers(x)