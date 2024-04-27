import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Batch


def _adj_to_edge_index(adj):
    # adj is [batch, num_nodes, num_nodes]
    edge_index = adj.nonzero(as_tuple=False).t().contiguous()
    return edge_index


class BatchGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.6):
        super().__init__()
        self.gat_conv = GATConv(
            in_dim, out_dim, heads=num_heads, concat=True, dropout=dropout)

    def forward(self, x, node_matrix):
        # x: Node features [batch, num_nodes, time, num_features]
        # node_matrix: [batch, num_nodes, num_nodes]
        edge_index = _adj_to_edge_index(node_matrix)
        for i in range(x.size(0)):
            for j in range(x.size(2)):
                x_t = x[i, :, j, :]
                edge_index_t = edge_index[:, edge_index[0] == i]

        pass
