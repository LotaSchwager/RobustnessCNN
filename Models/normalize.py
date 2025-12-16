import torch
import torch.nn as nn

class NormalizeLayer(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1,3,1,1))
        self.register_buffer("std", torch.tensor(std).view(1,3,1,1))

    def forward(self, x):
        return (x - self.mean) / self.std