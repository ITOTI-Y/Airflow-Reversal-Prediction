import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
from torch_geometric.nn.models import GAT
from ..config import CALCULATE_CONFIG

ENV_CONFIG = CALCULATE_CONFIG()


def adj_to_edge_index(adj): 
    """Convert an adjacency matrix to edge indices. 

    Args:
        adj (Tensor):[batch, num_nodes, num_nodes]
    Returns:
        edge_index(Tensor):[2, batch * num_edges]
    """    
    edge_index = adj.nonzero(as_tuple=False).T.contiguous()
    return edge_index


def edge_index_to_adj(x, edge_index):
    """Convert edge indices and edge features to an adjacency matrix.

    Args:
        x (Tensor):[Multiple batch nodes, times, nodes_features]
        edge_index (Tensor):[2, batch * num_edges]

    Returns:
        adj (Tensor):[num_nodes, num_nodes, nodes_features]
    """    
    adj = torch.zeros((edge_index.max().item()+1,
                      edge_index.max().item()+1, x.size(2)))
    for i in range(edge_index.size(1)):
        adj[edge_index[0, i], edge_index[1, i]] = x[i]
    adj = torch.mean(adj, dim=2)
    return adj


class BatchGATLayer(nn.Module):
    def __init__(self, in_dim, d_model, num_heads, dropout=0.6):
        """
        A batch processing layer based on Graph Attention Network (GAT).

        Args:
            in_dim (int): The dimension of input features
            d_model (int): The dimension of output features
            num_heads (int): The number of attention heads.If concat == True , 
                    Multi-head attention will be achieved through stacking operations. 
                    Else average operation will be performed.
            dropout (float, optional): The dropout rate to prevent overfitting. Defaults to 0.6.
        """
        super().__init__()
        self.gat_conv = GATConv(
            in_channels=in_dim, out_channels=d_model, heads=num_heads, concat=False, dropout=dropout)

    def forward(self, x, node_matrix, weights=None):
        """Forward pass of the BatchGATLayer.

        Args:
            x : Node features. [Multiple batch nodes, time, num_features]
            node_matrix : The adjacency matrix of the graph. [Multiple batch nodes, Multiple batch nodes]
            weights (bool, optional): If True, return the attention weights along with the output. Defaults to None.

        Returns:
            If weights is True:  
                Tuple of (attention_weights, edge_index)  
                attention_weights: Tensor of shape (batch_size, num_heads, num_nodes, num_nodes)  
                                   Containing the attention weights for each head and each batch.  
                edge_index: LongTensor of shape (2, num_edges)  
                            Containing the edge indices in COO format.  
            If weights is False:  
                Tensor of shape (batch_size, num_nodes, out_channels * heads (if concat=True) or out_channels (if concat=False))  
                Containing the updated node features for each batch.
        """        

        edge_index = adj_to_edge_index(node_matrix) # 
        if weights:
            # out: [edge_index, time, num_features]
            out = [self.gat_conv(x[:, i, :].float(), edge_index.long(
            ), return_attention_weights=weights)[1][1] for i in range(x.size(1))]
            return torch.stack(out, dim=1), edge_index
        else:
            out = [self.gat_conv(x[:, i, :].float(), edge_index.long(),)
                   for i in range(x.size(1))]
            return torch.stack(out, dim=0).transpose(1, 0)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout=0.1, max_time: int = 1800):
        """Initialize the PositionalEncoding class.

        Args:
            d_model (int): Dimension of the model input and output.
            dropout (float, optional): Dropout rate. Default is 0.1.
            max_time (int, optional): Maximum number of time steps (or positions) in the sequence. Defaults to 1800.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_time).unsqueeze(1)  # [max_time, 1]
        div_term = torch.exp(torch.arange(
            0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))  # [d_model//2]
        pe = torch.zeros(max_time, d_model)
        # Fill even indices with sine and odd indices with cosine functions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_time, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to the input data x
        # Since the first dimension of pe is 1, it will broadcast to match the batch size of x
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Temporal_Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.6, **kwargs):   
        """Initialize the Transformer with given parameters

        Args:
            d_model : Dimensionality of the model input and output
            nhead : Number of heads for multi-head attention
            num_encoder_layers : Number of encoder layers
            num_decoder_layers : Number of decoder layers
            dim_feedforward : Dimensionality of the feedforward network model in nn.TransformerEncoderLayer and nn.TransformerDecoderLayer
            dropout (float, optional): Dropout value to drop out on layers and attention probabilities. Defaults to 0.6.
        """        
        super().__init__()
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout, **kwargs)

    def forward(self, x):
        # x: [batch, num_nodes, time, num_features]
        batch_nodes, time_steps, num_features = x.size()
        src = x
        tgt = x[:, -1:, :]
        out = self.transformer(src, tgt)
        return out


class PINNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, size):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, size)
        pass

    def _people_exhaled(self, people:torch.Tensor , size:torch.Tensor) -> torch.Tensor:
        # Calculates the exhaled gas volume for each person
        result = torch.zeros_like(people)
        for i, value in enumerate(people):
            # Calculates the exhaled volume based on the number of people and room size
            result[i] = ENV_CONFIG.HUMAN_EXHALATION_FLOW * value.item()/size[i].item()
        return result
    
    def _flow_conv(self, flow):
        # Processes the flow data with a convolutional layer
        flow = flow.unsqueeze(0)
        flow = flow.permute(0, 3, 1, 2)
        # x: [1, num_features, edge_index, time]
        flow = self.conv(flow)  # Conv features with 1x1 kernel
        flow = flow.permute(0, 2, 3, 1).squeeze(0)
        # x: [1, edge_index, time, num_features]
        return flow

    def forward(self, origin_data, flow, edge_index):
        # origin_data: [Multiple batch nodes, time, num_features]
        # x: [edge_index, time, num_features]
        # edge_index: [2, edge_index]
        # result: [Multiple batch nodes, num_features]
        # time_step means the time step of the concentration monitoring step
        node_list = torch.zeros_like(edge_index.unique(
        ), device=flow.device, dtype=torch.float32, requires_grad=True)
        concentration = origin_data[:, -1, 0]
        mask = torch.ones_like(concentration)
        mask[-1] = 0
        size = origin_data[:, -1, 2]
        people = origin_data[:, -1, 1]
        flow = self._flow_conv(flow)
        flow_clone = flow.clone()
        node_list_new = node_list.clone()  # Avoid disrupting the computation graph.
        for _, (conn, value) in enumerate(zip(edge_index.T, flow)):
            if conn[0] != conn[1]:
                node_list_new[conn[0]] -= value.item()*concentration[conn[0]
                                                                     ].item()/size[conn[0]].item() * ENV_CONFIG.TIME_STEP
                node_list_new[conn[1]] += value.item()*concentration[conn[0]
                                                                     ].item()/size[conn[1]].item() * ENV_CONFIG.TIME_STEP
        result = concentration + (node_list_new + \
            self._people_exhaled(people, size) * ENV_CONFIG.TIME_STEP) * mask
        return result.unsqueeze(1), torch.cat([edge_index.T, flow[:, :, 0]], axis=1)


class Temporal_GAT_Transformer(nn.Module):
    def __init__(self, in_dim, d_model, num_heads, num_layers, dropout=0.1):
        super().__init__()

        # Initialize a Batch Graph Attention (GAT) Layer for feature extraction
        # input shape : [Multiple batch nodes, time, num_features]
        self.gat = BatchGATLayer(in_dim, d_model, num_layers, dropout)

        # Initialize a Positional Encoding layer to add positional information to the input sequence
        # input shape : [Multiple batch nodes, time, num_features]
        self.positon_encoding = PositionalEncoding(
            d_model=d_model, dropout=dropout)
        # Initialize a Temporal Transformer layer for sequence modeling
        # input shape : [Multiple batch nodes, time, num_features]
        self.transformer = Temporal_Transformer(d_model=d_model, nhead=num_heads, num_encoder_layers=num_layers,
                                                num_decoder_layers=num_layers, dim_feedforward=16, dropout=dropout, batch_first=True)
        # Initialize another Batch GAT Layer for final feature extraction and aggregation
        # input: [Multiple batch nodes, time, num_features]
        self.weights = BatchGATLayer(d_model, 1, d_model, dropout)

        # Initialize a PINN (Physics-Informed Neural Network) Layer for potentially incorporating physical laws or constraints
        # input: [Edge_index, time, num_features]
        self.pinn = PINNLayer(d_model, 1, 1)
        # output:
        # x: [Multiple batch nodes, num_features]
        # flow: [Node1, Node2, num_features]

    def forward(self, feature, node_matrix):
        # feature: Input feature matrix.Shape:[batch_size * num_nodes, time_steps, num_features]
        # node_matrix: Node matrix potentially containing relational or attribute information among nodes

        x = self.gat(feature, node_matrix)  # Process the input features and node matrix through the Graph Attention Network (GAT) layer
        x = self.positon_encoding(x)  # Add positional encoding information to the processed features
        x = self.transformer(x)  # Feed the positionally encoded features into the Temporal Transformer layer for further processing
        flow, edge_index = self.weights(
            x, node_matrix, weights=True)  # GAT attention weights get flow information
        x, flow = self.pinn(feature, flow, edge_index)
        return x, flow
