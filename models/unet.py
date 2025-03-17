import torch
import torch.nn as nn

class Unet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.tensor):
        raise NotImplementedError()
