import pathlib
import sys
from torch.utils.data import Dataset
sys.path.append(str(pathlib.Path(__file__).parents[1]))
from deepl.prep import prep_data


class NodeDataset(Dataset):
    def __init__(self, data_path:str=None):
        if data_path is not None:
            self.data_path = pathlib.Path(data_path)
        else:
            self.data_path = pathlib.Path(__file__).parents[2].joinpath('data')
        self.feature = []
        self.labels = []
        self.connection_matrices = []
        for file_path in self.data_path.glob('*.csv'):
            if "_matrix" in file_path.name:
                pass
            else:
                data = prep_data(file_path)
                self.feature.extend(data.values)

if __name__ == '__main__':
    print(NodeDataset().data_path)