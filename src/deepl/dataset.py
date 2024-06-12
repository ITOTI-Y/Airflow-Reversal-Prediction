import pathlib
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
from .prep import prep_data
from ..utils.flow import *
from ..config import CALCULATE_CONFIG

ENV_CONFIG = CALCULATE_CONFIG()

class NodeDataset(Dataset):
    def __init__(self, data_path:str=None, device=None):
        if data_path is not None:
            self.data_path = pathlib.Path(data_path)
        else:
            self.data_path = pathlib.Path(__file__).parents[2].joinpath('data')
        self.device = device
        self.feature, self.labels, self.connection_matrices, self.flow = self._build_dataset()
        self.keys = list(self.feature.keys())

    def _build_dataset(self):
        feature = {}
        labels = {}
        flow = {}
        node_matrix = {}
        for file_path in self.data_path.glob('*.csv'):
            if "_matrix" in file_path.name:
                building_identifier,values = prep_data(file_path)
                node_matrix[building_identifier] = torch.from_numpy(values).to(self.device)
            elif "_flow" in file_path.name:
                building_identifier, values = prep_data(file_path)
                values = compute_net_flow(values)
                flow[building_identifier] = values.to(self.device)
            else:
                building_identifier, values, label = prep_data(file_path)
                feature[building_identifier] = torch.from_numpy(values).to(self.device)
                labels[building_identifier] = torch.from_numpy(label).to(self.device)
        return feature, labels, node_matrix, flow
    
    def __len__(self):
        return len(self.feature)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        return self.feature[key], self.labels[key], self.connection_matrices[key], self.flow[key]
    
class NodeDataLoader(DataLoader):
    def __init__(self, dataset, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)
        self.collate_fn = self._collate_fn
        self.device = dataset.device
    
    def _collate_fn(self, batch):
        feature, label, node_matrix, flow = zip(*batch)
        feature = torch.cat([i for i in feature],axis=0)
        label = torch.cat([i for i in label],axis=0)
        label,feature = self._conbine_outside_node(label=label, feature=feature)
        node_matrix = self._combine_node_connection(node_matrix)
        flow = self._combine_flow(flow)
        return feature, label, node_matrix, flow
    
    def _combine_node_connection(self, node_matries:list | tuple):
        matrix_size = 0
        for i in node_matries:
            matrix_size += i.size(0)-1
        combined_matrix = torch.zeros(matrix_size+1, matrix_size+1).to(self.device)

        start = 0
        for node_matrix in node_matries:
            end = start + node_matrix.size(0)-1
            combined_matrix[start:end, start:end] += node_matrix[:-1,:-1]
            start = end
        outdoor_connection = torch.cat([matrix[:-1,-1] for matrix in node_matries])
        outdoor_connection = torch.cat([outdoor_connection, torch.ones(1).to(self.device)])
        combined_matrix[-1,:] += outdoor_connection
        combined_matrix[:,-1] += outdoor_connection
        return combined_matrix
    
    def _combine_flow(self, flows:list | tuple):
        start = 0
        for i in flows:
            lenght = i[:,:-1].max().item() + 1
            i[:,:-1] += start
            start += lenght
        temp = torch.cat(flows,dim=0)
        temp[temp.isinf()] = temp[:,:-1].max() + 1
        for t in temp:
            node1, node2, value = t[0].item(), t[1].item(), t[2].item()
            if t[0] > t[1]:
                t[0], t[1] = node2, node1
                t[2] = -value
        temp = pd.DataFrame(temp.cpu()).sort_values(by=[0,1]).values
        return torch.tensor(temp).to(self.device)
    
    def _conbine_outside_node(self, label:torch.Tensor, feature:torch.Tensor,) -> torch.Tensor:
        outside_node_feature = torch.ones((1,feature.size(1),feature.size(2))) * torch.tensor([ENV_CONFIG.OUTSIDE_CONCENTRATION,-1,-1])
        outside_node_label = torch.ones((1,label.size(1))) * torch.tensor([ENV_CONFIG.OUTSIDE_CONCENTRATION])
        return torch.cat([label, outside_node_label.to(self.device)], axis=0),torch.cat([feature, outside_node_feature.to(self.device)], axis=0)

    
if __name__ == '__main__':
    dataset = NodeDataset()
    print(dataset[0])