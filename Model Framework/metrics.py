import numpy as np
import torch
import torch.nn.functional as F

def mae_score(output, target):
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    return (1/output.shape[0]) * np.sum(np.abs(target - output))

def male_score(output, target):
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    
    return (1/output.shape[0]) * np.sum(np.abs(np.log(target+1) - np.log(output+1)))