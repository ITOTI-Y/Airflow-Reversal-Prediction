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
    
    def _collate_fn(self, batch):
        feature, label, connection_matrix = zip(*batch)
        feature = pad_sequence(feature, batch_first=True)
        label = pad_sequence(label, batch_first=True)
        connection_matrix = pad_sequence(connection_matrix, batch_first=True)
        return feature, label, connection_matrix
    
    def _pad_tensors_to_max_size(self, tensor_list:list):
        max_size = max([tensor.size(0) for tensor in tensor_list])
        padded_tensors = []
        for tensor in tensor_list:
            pad_size = max_size - tensor.size(0)
            pad = torch.zeros(pad_size, tensor.size(1))
            padded_tensor = torch.cat([tensor, pad], dim=0)
            padded_tensors.append(padded_tensor)
        return padded_tensors

if __name__ == '__main__':
    dataset = NodeDataset()
    print(dataset[0])