import torch
from torch.utils.data import DataLoader
from .dataset import NodeDataset, NodeDataLoader
from .model import *

class Train():
    def __init__(self, data_path:str=None):
        self.device = torch.device('cuda' if torch.cuda else 'cpu')
        self.dataset = NodeDataset(data_path, device=self.device)
        self.dataloader = NodeDataLoader(self.dataset, batch_size=2, shuffle=True)
    def train(self):
        model = BatchGATLayer(2, 2, 2).to(self.device)
        for feature, label, connection_matrix in self.dataloader:
            feature = feature.to(self.device)
            label = label.to(self.device)
            connection_matrix = connection_matrix.to(self.device)
            model(feature, connection_matrix)
            break