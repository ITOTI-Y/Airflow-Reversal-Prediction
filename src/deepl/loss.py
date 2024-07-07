import torch.nn as nn
import torch

class Combine_loss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = MSE_loss()
        self.mass = Mass_conservation_loss()
        self.log_vars = nn.Parameter(torch.zeros(2))
    
    def forward(self, pred, target, flow, c_flow):
        imbalance = self.mass(flow)
        mse = self.mse(pred, target)

        # 权重
        weight_imbalance = 10
        weight_mse = 1

        loss = weight_imbalance * imbalance + weight_mse * mse

        print(f'imbalance: {weight_imbalance * imbalance}, mse: {weight_mse * mse}, loss: {loss}')

        # 计算每个损失项的比例
        with torch.no_grad():
            imbalance_ratio = (weight_imbalance * imbalance) / loss
            mse_ratio = (weight_mse * mse) / loss
        
        return loss, imbalance_ratio.item(), mse_ratio.item()
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
        result = torch.zeros(flow.size(0))
        for i in range(flow.size(0)):
            result[i] = torch.sum(flow[i,:]) - torch.sum(flow[:,i])
        return torch.sum(torch.abs(result))