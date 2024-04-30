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
        self.feature, self.labels, self.connection_matrices = self._build_dataset()
        self.keys = list(self.feature.keys())
        pass

    def _build_dataset(self):
        feature = {}
        labels = {}
        connection_matrices = {}
        for file_path in self.data_path.glob('*.csv'):
            if "_matrix" in file_path.name:
                building_identifier,values = prep_data(file_path)
                connection_matrices[building_identifier] = torch.from_numpy(values).to(self.device)
            else:
                building_identifier, values, label = prep_data(file_path)
                feature[building_identifier] = torch.from_numpy(values).to(self.device)
                labels[building_identifier] = torch.from_numpy(label).to(self.device)
        return feature, labels, connection_matrices
    
    def __len__(self):
        return len(self.feature)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        return self.feature[key], self.labels[key], self.connection_matrices[key]
    
class NodeDataLoader(DataLoader):
    def __init__(self, dataset, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)
        self.collate_fn = self._collate_fn
        self.device = dataset.device
    
    def _collate_fn(self, batch):
        feature, label, node_matrix = zip(*batch)
        feature = torch.cat([i for i in feature],axis=0)
        label = torch.cat([i for i in label],axis=0)
        node_matrix = self._combine_node_connection(node_matrix)
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
    
    # def _pad_tensors_to_max_size(self, tensor_list:list):
    #     max_size = max([tensor.size(0) for tensor in tensor_list])
    #     padded_tensors = []
    #     for tensor in tensor_list:
    #         pad_tensor = torch.zeros(max_size, max_size).to(self.device)
    #         pad_tensor[:tensor.size(0), :tensor.size(1)] += tensor
    #         padded_tensors.append(pad_tensor)
    #     padded_tensors = torch.stack(padded_tensors)
    #     return padded_tensors

if __name__ == '__main__':
    dataset = NodeDataset()
    print(dataset[0])