import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .dataset import NodeDataset, NodeDataLoader
from .model import *
from .loss import *
from ..utils.flow import *

class Train():
    def __init__(self, data_path:str=None):
        self.device = torch.device('cuda' if torch.cuda else 'cpu')
        self.dataset = NodeDataset(data_path, device=self.device)
        self.dataloader = NodeDataLoader(self.dataset, batch_size=1, shuffle=True)
    def train(self,epochs:int=1000):
        loss = Combine_loss()
        model = Temporal_GAT_Transformer(in_dim=3, d_model=12, num_heads=6, num_layers=3).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
        for epoch in tqdm(range(epochs)):
            loss_value = self._step_train(model, optimizer, loss)
            print(f'Epoch: {epoch}, Loss: {loss_value}')

    def _step_train(self, model, optimizer, loss):
        model.train()
        for feature, label, connection_matrix, c_flow in self.dataloader:
            feature = feature.to(self.device)
            label = label.to(self.device)
            connection_matrix = connection_matrix.to(self.device)
            out, flow = model(feature, connection_matrix)
            c_flow = flows_to_matrix(c_flow)
            flow = flows_to_matrix(flow)
            loss_value, imbalance_ratio, mse_ratio = loss(out, label, flow, c_flow)
            loss_value.backward()
            optimizer.step()
        return loss_value.item()