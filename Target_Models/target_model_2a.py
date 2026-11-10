import torch.nn as nn
import torch
import torch.nn.functional as F

class TargetModel_2a(nn.Module):
    def __init__(self, input_size, input_channels=3, output_size=10):
        super(TargetModel_2a, self).__init__()
        # Block 1: 3 -> 16 φίλτρα
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2) # Μειώνει το 32x32 σε 16x16
        
        # Block 2: 16 -> 32 φίλτρα
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # Το επόμενο MaxPool θα μειώσει το 16x16 σε 8x8

        # Fully Connected Layer
        # Μετά από δύο MaxPool (32/2/2 = 8), η εικόνα είναι 8x8.
        # Άρα: 32 φίλτρα * 8 * 8 = 2048 χαρακτηριστικά
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # Output: 16x16x16
        x = self.pool(F.relu(self.conv2(x))) # Output: 32x8x8
        
        x = x.view(-1, 32 * 8 * 8) # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x