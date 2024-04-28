import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch


class BatchGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.6):
        """
        Apply GAT to each time step of a sequence of node features

        Args:
            in_dim (_type_): Node feature dimension
            out_dim (_type_): Node feature dimension
            num_heads (_type_): If concat == True , 
                    Multi-head attention will be achieved through stacking operations. 
                    Else average operation will be performed.
            dropout (float, optional): _description_. Defaults to 0.6.
        """
        super().__init__()
        self.gat_conv = GATConv(
            in_dim, out_dim, heads=num_heads, concat=False, dropout=dropout)
        
    def _adj_to_edge_index(self,adj):
        # adj is [batch, num_nodes, num_nodes]
        # edge_index is [2, batch * num_edges]
        edge_index = adj.nonzero(as_tuple=False).T[1:].contiguous()
        return edge_index

    def forward(self, x, node_matrix):
        # x: Node features [batch, num_nodes, time, num_features]
        # node_matrix: [batch, num_nodes, num_nodes]
        edge_index = self._adj_to_edge_index(node_matrix)
        batch_size, num_nodes, time_steps, num_features = x.size()
        # combine batch and node dimensions
        x = x.view(batch_size * num_nodes, time_steps, num_features)

        # apply GAT to each time step
        gat_out = []
        for t in range(time_steps):
            # out: [batch*num_nodes, num_features*num_heads]
            x_t = x[:, t, :]
            out = self.gat_conv(x_t.float(), edge_index.long())
            gat_out.append(out.unsqueeze(1))

        # output [batch, num_nodes, time, num_features*num_heads]
        return torch.cat(gat_out, dim=1).view(batch_size, num_nodes, time_steps, -1)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, dropout=0.6, max_time:int=1800):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_time).unsqueeze(1) # [max_time, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model)) # [d_model//2]
        pe = torch.zeros(max_time, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).unsqueeze(0) # [1, 1, max_time, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :, :x.size(2)]
        return self.dropout(x)


# class Temporal_Encoder_Layer(nn.Module):
#     def __init__(self, in_dim, num_heads, num_layers, dropout=0.6):
#         super().__init__()
#         self.encoder_layer = nn.TransformerEncoderLayer(
#             d_model=in_dim, nhead=num_heads, dropout=dropout)
#         self.transformer_encoder = nn.TransformerEncoder(
#             self.encoder_layer, num_layers=num_layers)

#     def forward(self, x):
#         # x: [batch, num_nodes, time, num_features]
#         # combine batch and node dimensions for transformer
#         batch_size, num_nodes, time_steps, num_features = x.size()
#         x = x.view(batch_size, num_nodes*time_steps, num_features)
#         x = self.transformer_encoder(x)
#         return x.view(batch_size, num_nodes, time_steps, num_features)


# class Temporal_Decoder_Layer(nn.Module):
#     def __init__(self, in_dim, num_heads, num_layers, dropout=0.6):
#         super().__init__()
#         self.decoder_layer = nn.TransformerDecoderLayer(
#             d_model=in_dim, nhead=num_heads, dropout=dropout)
#         self.transformer_decoder = nn.TransformerDecoder(
#             self.decoder_layer, num_layers=num_layers)

#     def forward(self, x):
#         # x: [batch, num_nodes, time, num_features]
#         # combine batch and node dimensions for transformer
#         batch_size, num_nodes, time_steps, num_features = x.size()
#         tgt = x[:, :, -1, :]
#         x = x.view(batch_size, num_nodes*time_steps, num_features)
#         out = self.transformer_decoder(tgt=tgt, memory=x)
#         return out.view(batch_size, num_nodes, time_steps, num_features)


class Temporal_Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.6, **kwargs):
        super().__init__()
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout, **kwargs)

    def forward(self, x):
        # x: [batch, num_nodes, time, num_features]
        batch_size, num_nodes, time_steps, num_features = x.size()
        src = x.view(batch_size, num_nodes*time_steps, num_features)
        tgt = x[:, :, -1, :]
        out = self.transformer(src, tgt)
        return out.view(batch_size, num_nodes, -1, num_features)
    
class Full_Connected(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        return self.fc(x)

class Temporal_GAT_Transformer(nn.Module):
    def __init__(self, in_dim, d_model, num_heads, num_layers, dropout=0.6):
        super().__init__()
        # input: [batch, num_nodes, time, num_features]
        self.gat = BatchGATLayer(in_dim, d_model, num_heads, dropout)
        # input: [batch, num_nodes, time, num_features]
        self.positon_encoding = PositionalEncoding(d_model)
        # input: [batch, num_nodes, time, num_features]
        self.transformer = Temporal_Transformer(d_model=d_model, nhead=num_heads, num_encoder_layers=num_layers,
                                                num_decoder_layers=num_layers, dim_feedforward=16, dropout=dropout, batch_first=True)
        # input: [batch, num_nodes, time, num_features]
        self.fc = Full_Connected(d_model, 1)

    def forward(self, x, node_matrix):
        x = self.gat(x, node_matrix)
        x = self.positon_encoding(x)
        x = self.transformer(x)
        x = self.fc(x) # 这里得到的应该是流量信息
        return x
