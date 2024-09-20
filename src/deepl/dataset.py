import pathlib
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple
from .prep import prep_data
from ..utils.flow import *
from ..config import CALCULATE_CONFIG

ENV_CONFIG = CALCULATE_CONFIG()

class NodeDataset(Dataset):
    def __init__(self, data_path:str=None, device=None):
        """Initializes an instance of the NodeDataset class.

        This function provides two choices of data_path creating, and it also define the device to perform.

        Args:
            data_path (str, optional): The path to the dataset.   
            device (str, optional): The device to perform computations on. 
        """     
        if data_path is not None:
            self.data_path = pathlib.Path(data_path)
        else:
            self.data_path = pathlib.Path(__file__).parents[2].joinpath('data')
        self.device = device
        self.feature, self.labels, self.connection_matrices, self.flow = self._build_dataset()
        self.keys = list(self.feature.keys())

    def _build_dataset(self):
        """Builds a dataset from CSV files in a specified directory.
        The files are categorized based on their names (suffixes) and processed accordingly.

        Returns:
            Tuple containing:
                feature (dict): Dictionary of features for each building.
                labels (dict): Dictionary of labels for each building.
                node_matrix (dict): Dictionary of node matrices for each building.
                flow (dict): Dictionary of flow data for each building.
        """
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
                values = self._compute_net_flow(values)
                flow[building_identifier] = values.to(self.device)
            else:
                building_identifier, values, label = prep_data(file_path)
                feature[building_identifier] = torch.from_numpy(values).to(self.device)
                labels[building_identifier] = torch.from_numpy(label).to(self.device)
        return feature, labels, node_matrix, flow
    
    def _compute_net_flow(self, flows:torch.Tensor) -> torch.Tensor:
        """Compute the net flow between nodes in a network given a tensor of flow records.
  
        Args:
            flows (torch.Tensor): A 2D tensor of shape (N, 3) where each row represents a flow record with [node1, node2, value] indicating the start node, end node, and the flow value respectively.  
  
        Returns:  
            torch.Tensor: A 2D tensor of shape (M, 3) where M <= N, representing the net flow between unique node pairs. Each row contains [node1, node2, net_flow_value].  
        """  
        flow_dict = {}
        if not isinstance(flows, torch.Tensor):
            flows = torch.tensor(flows)
        for flow in flows:
            node1, node2, value = flow[0].item(), flow[1].item(), flow[2]
            if node1 == node2:
                continue
            elif (node1,node2) in flow_dict:
                flow_dict[(node1,node2)] += value.clone()
            elif (node2,node1) in flow_dict:
                flow_dict[(node2,node1)] -= value.clone()
            else:
                flow_dict[(node1,node2)] = value.clone()
        result = []
        for i, (node1,node2) in enumerate(flow_dict.keys()):
            result.append(torch.tensor((node1, node2, flow_dict[(node1,node2)])))
        return torch.stack(result).to(flows.device)
    
    
    def __len__(self):
        """Return the number of data
        
        Returns:
            len(self.feature): The number of data
        """
        return len(self.feature)
    
    def __getitem__(self, idx):
        """Get an item from the dataset by its index.
    
        Args:
            idx (int): The index of the item to retrieve.
    
        Returns:
            tuple: A tuple containing:
                - feature
                - labels
                - connection_matrices
                - flow
        """
        key = self.keys[idx]
        return self.feature[key], self.labels[key], self.connection_matrices[key], self.flow[key]
    
class NodeDataLoader(DataLoader):
    def __init__(self, dataset, *args, **kwargs):
        """Initialize the instance of NodeDataLoader class.

        This method initializes the NodeDataLoader by calling the parent DataLoader's __init__,
        setting a custom collation function, and storing the dataset's device.

        Args:
            dataset: The dataset to load data from.
            *args: Variable length argument list to pass to the parent DataLoader.
            **kwargs: Arbitrary keyword arguments to pass to the parent DataLoader.

        Note:
            - Sets the `collate_fn` attribute to a custom `_collate_fn` method.
            - Sets the `device` attribute to the device specified in the dataset.

        """
        super().__init__(dataset, *args, **kwargs)
        self.collate_fn = self._collate_fn
        self.device = dataset.device
    
    def _collate_fn(self, batch):
        """Collates and processes batch data for the DataLoader.
    
        This method unpacks the batch, concatenates features and labels,
        processes the data, and combines node connections and flow data.
    
        Args:
            batch: A list of tuples, where each tuple contains feature, label,
                   node_matrix, and flow data for a single sample.
    
        Returns:
            tuple: A tuple containing:
                - feature (torch.Tensor): Concatenated and processed feature data.
                - label (torch.Tensor): Concatenated and processed label data.
                - node_matrix: Combined node connection matrices.
                - flow: Combined flow data.
        """
        feature, label, node_matrix, flow = zip(*batch)
        feature = torch.cat([i for i in feature],axis=0)
        label = torch.cat([i for i in label],axis=0)
        label,feature = self._conbine_outside_node(label=label, feature=feature)
        node_matrix = self._combine_node_connection(node_matrix)
        flow = self._combine_flow(flow)
        return feature, label, node_matrix, flow
    
    def _combine_node_connection(self, node_matries:list | tuple):
        """Combines adjacency matrices of multiple graphs into a single larger matrix.
    
        This method combines multiple adjacency matrices into one, considering
        connections to the outdoor environment. It creates a new matrix where
        individual graphs are placed along the diagonal, and outdoor connections
        are represented in the last row and column.
    
        Args:
            node_matries: A list or tuple of adjacency matrices (torch.Tensor)
                for individual graphs. Each matrix should have an extra row and
                column for outdoor connections.
    
        Returns:
            torch.Tensor: The combined adjacency matrix, including outdoor connections.
    
        Raises:
            ValueError: If node_matries is empty or contains non-tensor elements.
        """
        matrix_size = 0
        for i in node_matries:
            matrix_size += i.size(0)-1
        combined_matrix = torch.zeros(matrix_size+1, matrix_size+1).to(self.device)

        # Iterate through all node connection matrices to calculate the size of the combined matrix.
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
    
    def _combine_flow(self, flows:List[torch.Tensor] | Tuple[torch.Tensor,]):
       """Combines multiple flow data sets into a single, unified flow representation.
    
        This method processes and combines multiple flow data tensors, typically representing
        edges in a directed graph. It ensures node indices are unique across the combined
        flow and handles the direction of edges.
    
        Args:
            flows: A list or tuple of flow data tensors. Each tensor is of shape (N, 3),
                where N is the number of edges, and each row contains 
                [start_node, end_node, edge_weight].
    
        Returns:
            torch.Tensor: A combined flow tensor of shape (M, 3), where M is the total
                number of edges across all input flows. Each row contains
                [start_node, end_node, edge_weight], where:
                - Node indices have been renumbered for uniqueness.
                - Edge directions have been standardized (start_node < end_node).
                - Rows are sorted by start_node, then end_node.
    
        Note:
            - Infinite values in the input are replaced with (max_node_index + 1).
            - The method uses pandas for sorting, which may impact performance for large datasets.
        """
        start = 0
        for i in flows:
            # Calculate the maximum node index in the current flow (based on edge start and end indices)
            lenght = i[:,:-1].max().item() + 1
            # Update the start and end node indices in the current flow to account for nodes in previous flows
            i[:,:-1] += start
            # Update the starting offset for the next flow
            start += lenght

        # Concatenate all flow data along the first dimension
        temp = torch.cat(flows,dim=0)

        # Handle infinite values (inf) in the concatenated data by replacing them with the maximum node index + 1 (potentially incorrect logic)
        temp[temp.isinf()] = temp[:,:-1].max() + 1

        # Iterate over each row (edge) in the concatenated data to make necessary adjustments
        for t in temp:
            node1, node2, value = t[0].item(), t[1].item(), t[2].item()
            if t[0] > t[1]:
                t[0], t[1] = node2, node1
                t[2] = -value
        
        # Convert the processed flow data to a pandas DataFrame to sort by start and end node indices, then back to a Tensor on the correct device
        temp = pd.DataFrame(temp.cpu()).sort_values(by=[0,1]).values
        return torch.tensor(temp).to(self.device)
    
    def _conbine_outside_node(self, label:torch.Tensor, feature:torch.Tensor,) -> Tuple[torch.Tensor,torch.Tensor]:
        """Adds features and labels for an outside (or virtual) node to the existing label and feature tensors.
    
        Args:
            label (torch.Tensor): The existing tensor of node labels. Shape: (N, L) where N is the number of nodes
                and L is the number of label dimensions.
            feature (torch.Tensor): The existing tensor of node features. Shape: (N, T, F) where N is the number of nodes,
                T is the number of time steps, and F is the number of feature dimensions.
    
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - Updated label tensor (including the outside node's label). Shape: (N+1, L)
                - Updated feature tensor (including the outside node's features). Shape: (N+1, T, F)
    
        """
        outside_node_feature = torch.ones((1,feature.size(1),feature.size(2))) * torch.tensor([ENV_CONFIG.OUTSIDE_CONCENTRATION,-1,-1])
        outside_node_label = torch.ones((1,label.size(1))) * torch.tensor([ENV_CONFIG.OUTSIDE_CONCENTRATION])
        return torch.cat([label, outside_node_label.to(self.device)], axis=0),torch.cat([feature, outside_node_feature.to(self.device)], axis=0)

    
if __name__ == '__main__':
    dataset = NodeDataset()
    print(dataset[0])
