import torch.nn as nn
import torch

class Combine_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = MSE_loss()
        self.mass = Mass_conservation_loss()
    
    def forward(self, pred, target, flow, c_flow):
        self.mass(flow)
        # return self.mse(pred, target)
    
class MSE_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, pred, target):
        return self.loss(pred, target)
    
class Mass_conservation_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, flow):
        nodes = {i.item():0 for i in torch.unique(flow[:,:-1])}
        for f in flow:
            nodes[f[0].item()] += f[-1].item()
            nodes[f[1].item()] -= f[-1].item()
        return torch.sum(torch.tensor(list(nodes.values())))