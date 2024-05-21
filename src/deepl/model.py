import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
from torch_geometric.nn.models import GAT


def adj_to_edge_index(adj):
    # adj is [batch, num_nodes, num_nodes]
    # edge_index is [2, batch * num_edges]
    edge_index = adj.nonzero(as_tuple=False).T.contiguous()
    return edge_index


def edge_index_to_adj(x, edge_index):
    # edge_index is [2, batch * num_edges]
    # x is [Multiple batch nodes, times, nodes_features]
    # adj is [num_nodes, num_nodes, nodes_features]
    adj = torch.zeros((edge_index.max().item()+1,
                      edge_index.max().item()+1, x.size(2)))
    for i in range(edge_index.size(1)):
        adj[edge_index[0, i], edge_index[1, i]] = x[i]
    adj = torch.mean(adj, dim=2)
    return adj


class BatchGATLayer(nn.Module):
    def __init__(self, in_dim, d_model, num_heads, dropout=0.6):
        """
        Apply GAT to each time step of a sequence of node features

        Args:
            in_dim (_type_): Node feature dimension
            d_model (_type_): Node feature dimension
            num_heads (_type_): If concat == True , 
                    Multi-head attention will be achieved through stacking operations. 
                    Else average operation will be performed.
            dropout (float, optional): _description_. Defaults to 0.6.
        """
        super().__init__()
        self.gat_conv = GATConv(
            in_channels=in_dim, out_channels=d_model, heads=num_heads, concat=False, dropout=dropout)

    def forward(self, x, node_matrix, weights=None):
        # x: Node features [Multiple batch nodes, time, num_features]
        # node_matrix: [Multiple batch nodes, Multiple batch nodes]
        edge_index = adj_to_edge_index(node_matrix)
        if weights:
            out = [self.gat_conv(x[:, i, :].float(), edge_index.long(
            ), return_attention_weights=weights)[1][1] for i in range(x.size(1))]
            return torch.stack(out, dim=1), edge_index
        else:
            out = [self.gat_conv(x[:, i, :].float(), edge_index.long(),)
                   for i in range(x.size(1))]
            return torch.stack(out, dim=0).transpose(1, 0)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout=0.1, max_time: int = 1800):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_time).unsqueeze(1)  # [max_time, 1]
        div_term = torch.exp(torch.arange(
            0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))  # [d_model//2]
        pe = torch.zeros(max_time, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_time, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Temporal_Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.6, **kwargs):
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

    def _people_exhaled(self, people, size):
        HUMAN_EXHALATION = 0.0001  # 人体潮气量 m^3/s
        HUMAN_EXHALATION_CONCENTRATION = 40000  # 人体潮气二氧化碳浓度 ppm
        HUMAN_EXHALATION_FLOW = HUMAN_EXHALATION * \
            HUMAN_EXHALATION_CONCENTRATION  # 人体潮气二氧化碳流量 ppm*m^3/s
        for i, value in enumerate(people):
            people[i] = HUMAN_EXHALATION_FLOW * value.item()/size[i].item()
        return people

    def forward(self, origin_data, x, edge_index):
        node_list = torch.zeros_like(edge_index.unique(),device=x.device, dtype=torch.float32, requires_grad=True)
        concentration = origin_data[:, -1, 0]
        size = origin_data[:, -1, 2]
        people = origin_data[:, -1, 1]
        x = x.unsqueeze(0)
        x = x.permute(0, 3, 1, 2)
        # x = x.permute()
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).squeeze(0)
        flow = x.clone()
        node_list_new = node_list.clone() # 避免破坏计算图
        for i, (conn, value) in enumerate(zip(edge_index.T, x)):
            if conn[0] != conn[1]:
                node_list_new[conn[0]] = value.item()*concentration[conn[0]
                                                                 ].item()/size[conn[0]].item()
                node_list_new[conn[1]] = value.item()*concentration[conn[0]
                                                                 ].item()/size[conn[1]].item()
        result = concentration + node_list_new + self._people_exhaled(people, size)
        return result.unsqueeze(1), flow


class Temporal_GAT_Transformer(nn.Module):
    def __init__(self, in_dim, d_model, num_heads, num_layers, dropout=0.6):
        super().__init__()
        # input: [Multiple batch nodes, time, num_features]
        # self.gat = BatchGATLayer(in_dim, d_model, num_heads, dropout)
        self.gat = BatchGATLayer(in_dim, d_model, num_layers, dropout)
        # input: [Multiple batch nodes, time, num_features]
        self.positon_encoding = PositionalEncoding(d_model)
        # input: [Multiple batch nodes, time, num_features]
        self.transformer = Temporal_Transformer(d_model=d_model, nhead=num_heads, num_encoder_layers=num_layers,
                                                num_decoder_layers=num_layers, dim_feedforward=16, dropout=dropout, batch_first=True)
        # input: [Multiple batch nodes, time, num_features]
        self.weights = BatchGATLayer(d_model, 1, d_model, dropout)
        # input: [Edge_index, time, num_features]
        self.pinn = PINNLayer(d_model, 1, 1)
        # output: 
        # x: [Multiple batch nodes, num_features]
        # flow: [Edge_index, num_features]

    def forward(self, origin_data, node_matrix):
        x = self.gat(origin_data, node_matrix)
        x = self.positon_encoding(x)
        x = self.transformer(x)
        x, edge_index = self.weights(
            x, node_matrix, weights=True) # GAT attention weights get flow information
        # x = edge_index_to_adj(x, edge_index)
        x, flow = self.pinn(origin_data, x, edge_index)
        return x, flow[:,:,0]
