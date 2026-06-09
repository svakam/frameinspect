import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class FrameAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder: compresses to a 1x64x64 bottleneck 
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), #1x64x64 -> 16x64x64
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 16x64x64 -> 16x32x32
            nn.Conv2d(16, 8, kernel_size=3, padding=1), # 16x32x32 -> 8x32x32
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # 8x32x32 -> 8x16x16
        )

        # decoder: expand bottleneck to original dim
        