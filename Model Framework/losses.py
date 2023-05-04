import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MeanAbsoluteError

__all__ = ['MALELoss']

# Mean Absolute Log Error
class MALELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mae = MeanAbsoluteError()
        
    def forward(self, output, target):
        return self.mae(torch.log(output+1), torch.log(target+1))