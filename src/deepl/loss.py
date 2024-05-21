import torch.nn as nn

class Combine_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = MSE_loss()
    
    def forward(self, pred, target):
        return self.mse(pred, target)
    
class MSE_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, pred, target):
        return self.loss(pred, target)