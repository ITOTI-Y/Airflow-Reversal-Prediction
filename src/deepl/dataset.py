import pathlib
import torch
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
from .prep import prep_data


class NodeDataset(Dataset):
    def __init__(self, data_path:str=None, device=None):
        if data_path is not None:
            self.data_path = pathlib.Path(data_path)
        else:
            self.data_path = pathlib.Path(__file__).parents[2].joinpath('data')
        self.device = device
        self.feature, self.labels, self.connection_matrices, self.flow = self._build_dataset()
        self.keys = list(self.feature.keys())
        pass

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
                flow[building_identifier] = None
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
        node_matrix = self._combine_node_connection(node_matrix)
        # flow = self._combine_flow(flow)
        return feature, label, node_matrix
    
    def _combine_node_connection(self, node_matries:list | tuple):
        matrix_size = 0
        for i in node_matries:
            matrix_size += i.size(0)
        combined_matrix = torch.zeros(matrix_size, matrix_size).to(self.device)

        start = 0
        for node_matrix in node_matries:
            end = start + node_matrix.size(0)
            combined_matrix[start:end, start:end] += node_matrix
            start = end
        return combined_matrix
    
    def _combine_flow(self, flow:list | tuple):
        index = 0
        start = 0
        combine_flow = torch.zeros_like(torch.cat([i for i in flow],axis=0))
        for f in flow:
            f[:,:-1] += index
            end = start + f.size(0)
            index += f[:,:-1].max().item() + 1
            combine_flow[start:end] += f
            start = end
        pass

if __name__ == '__main__':
    dataset = NodeDataset()
    print(dataset[0])