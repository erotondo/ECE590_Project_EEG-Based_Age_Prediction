import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

__all__ = ['CNNAgePrediction','LSTMAgePrediction']


class CNNAgePrediction(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(4, 16, (7, 64), (1,3))
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 64, (7, 16), (1,3))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d((14,266))
        self.lin = nn.Linear(64,1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print("Block 1")
        # print(out.shape)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # print("Block 2")
        # print(out.shape)

        out = self.pool(out)
        # print("After Pooling")
        # print(out.shape)
        out = torch.squeeze(out)
        out = self.lin(out)
        out = self.relu(out) # Force predictions to be non-negative

        return torch.squeeze(out)

class LSTMAgePrediction(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.num_layers = 2
        self.input_size = 26
        self.hidden_size = 128
        self.lstm = nn.LSTM(self.input_size,self.hidden_size,
                            self.num_layers,batch_first=True)
        self.relu = nn.ReLU()
        self.lin = nn.Linear(128,1)

    def forward(self, x):
        # Reshape for correct LSTM input
        x = torch.transpose(x, 1, 2)

        output, (h_n, c_n) = self.lstm(x)
        h_n = h_n[self.num_layers-1,:,:]
        h_n = torch.squeeze(h_n)
        #print(h_n.shape)

        out = self.relu(h_n)
        out = self.lin(out)
        out = self.relu(out) # Force predictions to be non-negative

        return torch.squeeze(out)