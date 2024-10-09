import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
from torch_geometric.nn.models import GAT
from ..config import CALCULATE_CONFIG

ENV_CONFIG = CALCULATE_CONFIG()


def adj_to_edge_index(adj): 
    """Converts an adjacency matrix to edge indices.

    This function transforms a given adjacency matrix into a sparse edge index
    format (COO format) that is widely used in graph neural networks.

    Args:
        adj (torch.Tensor): Adjacency matrix of shape [batch_size, num_nodes, num_nodes],
                            where 'batch_size' is the number of different graph instances
                            in the batch, 'num_nodes' is the total number of nodes in each graph.
                            Entries in this matrix are typically binary (1 if an edge exists between two
                            nodes, 0 otherwise).

    Returns:
        edge_index (torch.Tensor): A tensor of shape [2, num_edges] where each column represents an edge
                                   between two nodes in coordinate format (COO). 'num_edges' represents
                                   the total number of edges across all graphs in the batch.
    """
    edge_index = adj.nonzero(as_tuple=False).T.contiguous()
    return edge_index


def edge_index_to_adj(x, edge_index):
    """Converts edge indices and edge features to an adjacency matrix.

    Args:
        x (Tensor): A tensor of shape [batch_size*total_edges, features], where
                    each row represents the edge features for a specific edge in the batch.
        
        edge_index (Tensor): A tensor of shape [2, batch_size*num_edges] that 
                             contains the indices of the nodes connected by each edge. The first row 
                             corresponds to the source nodes and the second row to the target nodes.

    Returns:
        adj (Tensor): A 2D adjacency matrix of shape [num_nodes, num_nodes]
                      where each entry i, j represents the edge weight (mean of features) of 
                      the edge connecting node i to node j.
    """    
    # Initialize an empty adjacency matrix with size based on max node index
    adj = torch.zeros((edge_index.max().item() + 1,
                      edge_index.max().item() + 1, x.size(1)))
    
    # Populate adjacency matrix using edge information
    for i in range(edge_index.size(1)):
        adj[edge_index[0, i], edge_index[1, i]] = x[i]
    
    # Average all the feature dimensions to create a single valued adjacency matrix
    adj = torch.mean(adj, dim=2)
    
    return adj


class BatchGATLayer(nn.Module):
    def __init__(self, in_dim, d_model, num_heads, dropout=0.6):
        """
        Initializes a batch processing layer based on the Graph Attention Network (GAT).

        Args:
            in_dim (int): The dimension of the input feature vectors.
            d_model (int): The dimension of the output feature vectors for each node.
            num_heads (int): The number of attention heads used in the layer.
                If 'concat' is set to True, the output from all heads will be concatenated.
                Otherwise, their mean will be computed for the final output.
            dropout (float, optional): The dropout rate to mitigate the chance of overfitting.
                Defaults to 0.6.

        """
        super().__init__()
        self.gat_conv = GATConv(
            in_channels=in_dim, 
            out_channels=d_model, 
            heads=num_heads, 
            concat=False, 
            dropout=dropout
        )

    def forward(self, x, node_matrix, weights=None):
        """Forward pass of the BatchGATLayer.

        Args:
            x (Tensor): Node features. Shape of [batch_size * num_nodes, time_steps, num_features]
                        Where 'batch_size * num_nodes' is the total number of nodes in a batch.
            node_matrix (Tensor): The adjacency matrix of the graph. Shape of [batch_size * num_nodes, batch_size * num_nodes]
                                  representing connections between nodes.
            weights (bool, optional): If True, returns the attention weights along with the output. Defaults to None.

        Returns:
            If weights is True:
                Tuple:
                    - Tensor: attention_weights of shape `batch_size, num_heads, num_nodes, num_nodes`.
                              Contains attention weights calculated during the forward pass.
                    - LongTensor: edge_index of shape `2, num_edges`.
                                  Contains sparse tensor information in COO format, representing edges between nodes (source, target).
            
            If weights is False:
                Returns only the processed node features:
                    - Tensor: of shape `batch_size, num_nodes, out_channels * heads (if concat=True) or out_channels (if concat=False)`,
                              containing updated node embeddings after applying GAT convolution for each time_step.
        """        

        edge_index = adj_to_edge_index(node_matrix)  # Convert adjacency matrix to edge index in COO format.
        
        if weights:
            # Compute attention weights for each time step in the input sequence.
            # out: [batch_size, time_steps, num_heads, num_nodes, num_nodes]
            out = [self.gat_conv(
                       x[:, i, :].float(), 
                       edge_index.long(),
                       return_attention_weights=weights)[1][1] for i in range(x.size(1))
                  ]
            # Stack outputs along the second dimension (time_steps).
            return torch.stack(out, dim=1), edge_index
        
        else:
            # Apply GAT convolution at each time step (without returning attention weights).
            # Processed output shape: [batch_size, num_nodes, out_channels * heads]
            out = [self.gat_conv(x[:, i, :].float(), edge_index.long())
                   for i in range(x.size(1))]
            
            # Stack along 0th dimension and transpose to [batch_size, time_steps, ...].
            return torch.stack(out, dim=0).transpose(1, 0)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout=0.1, max_time: int = 1800):
        """Initializes the PositionalEncoding module.

        Args:
            d_model (int): The dimension of the model's input and output.
            dropout (float, optional): Dropout rate, a float value between 0 and 1. Controls the probability of an element being zeroed out for regularization purposes. Defaults to 0.1.
            max_time (int, optional): The maximum number of time steps (or positions) in the sequence. 
                                      This determines how long the positional encoding can be at maximum. Defaults to 1800.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encoding matrix. 
        # Create a tensor containing position indices, from 0 to max_time-1. The shape is [max_time, 1].
        position = torch.arange(0, max_time).unsqueeze(1)  # Shape: [max_time, 1]
        
        # Compute the scaling factor for positional encoding.
        # div_term helps to vary the positional encoding for every even and odd dimension differently.
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))  # [d_model//2]
        
        # Initialize the positional encoding tensor. Shape: [max_time, d_model]
        pe = torch.zeros(max_time, d_model)
        
        # Assign even-indexed dimensions to apply the sine function, and odd-indexed dimensions to apply the cosine function.
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices.
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices.
        
        # Add an extra dimension to the beginning, making the shape [1, max_time, d_model].
        # This ensures the encoding can be broadcasted across the batch dimension during forward pass.
        pe = pe.unsqueeze(0)  # Shape: [1, max_time, d_model]
        
        # Register 'pe' as a buffer so its content is not updated during training and it can be saved/loaded with the model.
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Forward pass of the PositionalEncoding module.
        
        Adds the positional encoding to the input and applies dropout.

        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, d_model]. Here, seq_len must be <= max_time.
        
        Returns:
            Tensor: The input tensor with positional encoding added and dropout applied.
        """
        # Add positional encodings to the input data 'x'. The encoding adjusts according to sequence length.
        # Since 'pe' has shape [1, max_time, d_model], broadcasting occurs automatically to match the size of 'x'.
        x = x + self.pe[:, :x.size(1), :]
        
        # Apply dropout to the output of the positional encoding.
        return self.dropout(x)


class Temporal_Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.6, **kwargs):   
        """Initialize the Temporal Transformer module.

        Args:
            d_model (int): Dimensionality of the inputs and outputs in the model. 
                           It typically represents the size of the feature embedding.
            nhead (int): The number of attention heads for the multi-head attention layers. 
                         More heads allow the model to attend to information from different perspectives.
            num_encoder_layers (int): Number of encoder layers (nn.TransformerEncoderLayer instances). Each layer applies 
                                      self-attention and feedforward operations.
            num_decoder_layers (int): Number of decoder layers (nn.TransformerDecoderLayer instances). In each layer, attention 
                                      is applied to both the encoder output and the current state of the decoder.
            dim_feedforward (int): The dimensionality of the feedforward network hidden layers 
                                   inside both encoder and decoder layers.
            dropout (float, optional): Dropout probability used in both attention and feedforward layers for 
                                       regularization. Defaults to 0.6.
            **kwargs: Additional keyword arguments for configuring the nn.Transformer module.
        """        
        super().__init__()
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout, **kwargs)

    def forward(self, x):
        """Defines the forward pass of the Temporal Transformer.

        The input tensor `x` is expected to be of shape [batch_size, num_nodes, time_steps, num_features].
        The transformer operates by attending to the input at all time steps and returns an output for the last time step.

        Args:
            x (Tensor): Input tensor containing data across multiple time steps. 
                        Shape: (batch_size, num_nodes, time_steps, num_features).

        Returns:
            Tensor: Output tensor after being processed by the transformer. It represents predictions at 
                    the last time step for each node. 
                    Shape: (batch_size, num_nodes, 1, num_features).
        """
        # Reshape the input tensor to accommodate the transformer layer
        batch_nodes, time_steps, num_features = x.size()

        # Source tensor (src) contains data from all time steps
        src = x
        
        # Target tensor (tgt) contains only the data at the last time step for each node
        tgt = x[:, -1:, :]
        
        # Forward pass through the transformer model
        out = self.transformer(src, tgt)
        return out


class PINNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, size):
        """Initializes the PINNLayer module.

        Args:
            in_dim (int): Number of input channels for the convolutional layer.
            out_dim (int): Number of output channels for the convolutional layer.
            size (int or tuple): Size of the convolutional kernel (height, width).
        """
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, size)

    def _people_exhaled(self, people: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
        """Calculates the human-exhaled gas volume per unit room size.

        This function computes the volume of CO2 exhaled by people in the given room 
        sizes. The results are scaled by the size of each node (room).

        Args:
            people (torch.Tensor): Tensor containing the number of people in each
                room (shape: [num_nodes]).
            size (torch.Tensor): Tensor with the size of each room (shape: [num_nodes]).

        Returns:
            torch.Tensor: Tensor containing the exhaled CO2 volume for each room, 
            normalized by the room size (shape: [num_nodes]).
        """
        result = torch.zeros_like(people)
        for i, value in enumerate(people):
            result[i] = ENV_CONFIG.HUMAN_EXHALATION_FLOW * value.item() / size[i].item()
        return result

    def _flow_conv(self, flow: torch.Tensor) -> torch.Tensor:
        """Applies a 2D convolution to the flow data.

        Performs a convolution operation on the flow tensor, which is first reshaped
        to match the required input format for the torch.nn.Conv2d layer.

        Args:
            flow (torch.Tensor): Tensor containing the flow values between nodes,
            with shape [edge_index, time, num_features].

        Returns:
            torch.Tensor: The transformed flow tensor with shape [edge_index, time, num_features],
            where its features have been processed via a 2D convolution.
        """
        flow = flow.unsqueeze(0)  # Add a batch-dimension: [1, edge_index, time, num_features]
        flow = flow.permute(0, 3, 1, 2)  # Rearrange to [1, num_features, edge_index, time]
        flow = self.conv(flow)  # Apply convolution on the flow data
        flow = flow.permute(0, 2, 3, 1).squeeze(0)  # Re-arrange back to [edge_index, time, num_features]
        return flow

    def forward(self, origin_data: torch.Tensor, flow: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute node concentrations based on flow and room properties.

        This method calculates the CO2 concentration for each node (room) based on 
        the incoming and outgoing flows from connected nodes and the exhaled gas from 
        people within the room.

        Args:
            origin_data (torch.Tensor): Input data containing node features over time, 
                with shape [num_nodes, time_steps, num_features]. The last time step 
                is used for this calculation.
            flow (torch.Tensor): Flow data between the nodes, indicating how air moves 
                from node to node, with shape [edge_index, time, num_features].
            edge_index (torch.Tensor): Tensor representing the edges between nodes, 
                with shape [2, num_edges].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Updated CO2 concentration of each node at the last timestep with shape [num_nodes, 1].
                - Tensor combining edge indices and flow data with shape [edge_index, 3].
        """
        node_list = torch.zeros_like(edge_index.unique(), device=flow.device, dtype=torch.float32, requires_grad=True)
        
        # Retrieve necessary features from the node input data
        concentration = origin_data[:, -1, 0]  # Concentration at the last time step
        mask = torch.ones_like(concentration)  # Mask for excluding external node (last node is external node)
        mask[-1] = 0
        size = origin_data[:, -1, 2]  # Room sizes at the last time step
        people = origin_data[:, -1, 1]  # Number of people at the last time step

        # Perform convolution on the flow data
        flow = self._flow_conv(flow)
        flow_clone = flow.clone()

        # Create a new node list tensor to hold flow-based updates to the CO2 concentrations
        node_list_new = node_list.clone()  # Clone to avoid disrupting the computation graph.

        # Update node concentrations based on flow and room properties
        for _, (conn, value) in enumerate(zip(edge_index.T, flow)):
            if conn[0] != conn[1]:  # Ignore self-connections
                node_list_new[conn[0]] -= value.item() * concentration[conn[0]].item() / size[conn[0]].item() * ENV_CONFIG.TIME_STEP
                node_list_new[conn[1]] += value.item() * concentration[conn[0]].item() / size[conn[1]].item() * ENV_CONFIG.TIME_STEP

        # Final concentration update: Includes air flow effects + human exhalation
        result = concentration + (node_list_new + self._people_exhaled(people, size) * ENV_CONFIG.TIME_STEP) * mask

        return result.unsqueeze(1), torch.cat([edge_index.T, flow[:, :, 0]], axis=1)


class Temporal_GAT_Transformer(nn.Module):
    """A model combining Graph Attention Networks (GAT) and Temporal Transformers.

    This model processes time sequences and node features using a combination of 
    Graph Attention Network (GAT) layers for structural data (e.g., node relationship)
    and a Transformer for temporal data. Additionally, physics-informed constraints are
    applied through a Physics-Informed Neural Network (PINN) layer.

    Attributes:
        gat (BatchGATLayer): The first batch-processing Graph Attention Network (GAT) 
                             layer which extracts features from node relationships.
        positon_encoding (PositionalEncoding): Encodes time as positional information 
                                               and adds it to the input tensor.
        transformer (Temporal_Transformer): Transformer that models sequences over time.
        weights (BatchGATLayer): A second GAT layer that outputs flow information across
                                 graph edges using attention weights.
        pinn (PINNLayer): Applies physics-informed neural network constraints on 
                          the flow between nodes (e.g., mass conservation, exhalation).
    """

    def __init__(self, in_dim, d_model, num_heads, num_layers, dropout=0.1):
        """Initializes a Temporal GAT-Transformer model.

        Args:
            in_dim (int): The input feature dimension for each node.
            d_model (int): The dimensionality of the feature embeddings.
            num_heads (int): The number of attention heads in the Transformer model.
            num_layers (int): The number of encoder and decoder layers in the Transformer.
            dropout (float, optional): The dropout rate used for regularization. Defaults to 0.1.
        """
        super().__init__()

        # Graph Attention Network (GAT) for feature extraction from nodes and edges
        self.gat = BatchGATLayer(in_dim, d_model, num_layers, dropout)

        # Positional encoding to add temporal context to the features
        self.positon_encoding = PositionalEncoding(d_model=d_model, dropout=dropout)

        # Transformer network to model sequences across time steps.
        self.transformer = Temporal_Transformer(d_model=d_model,
                                                nhead=num_heads,
                                                num_encoder_layers=num_layers,
                                                num_decoder_layers=num_layers,
                                                dim_feedforward=16,  # Hidden layer dimension size
                                                dropout=dropout,
                                                batch_first=True)

        # A GAT layer that computes attention weights and outputs the flow information
        self.weights = BatchGATLayer(d_model, 1, d_model, dropout)

        # Physics-Informed Neural Network (PINN) that applies physical constraints 
        # (like mass conservation) to the flow between nodes.
        self.pinn = PINNLayer(d_model, 1, 1)

    def forward(self, feature, node_matrix):
        """Performs a forward pass through the model.

        Args:
            feature (Tensor): An input feature matrix of shape 
                [batch_size * num_nodes, time_steps, num_features]. This contains 
                temporal features for each node.
            node_matrix (Tensor): A matrix representing node adjacency relationships. 
                Shape: [batch_size * num_nodes, batch_size * num_nodes].

        Returns:
            x (Tensor): Updated node embeddings for the last time step.
            flow (Tensor): Updated flow between nodes.
        """
        # Process input features through the first Graph Attention Network (GAT) layer.
        # Extracts contextualized feature embeddings based on node relationships.
        x = self.gat(feature, node_matrix)

        # Encode time series/positional information into the graph features.
        x = self.positon_encoding(x)

        # Process the embeddings over time using a Transformer, 
        # which captures temporal relationships between features.
        x = self.transformer(x)

        # Apply the second GAT layer to compute attention weights, 
        # which offer flow information between the nodes.
        flow, edge_index = self.weights(x, node_matrix, weights=True)

        # The PINN layer applies physics-informed constraints, updating the nodes' 
        # CO2 concentrations based on people, room size, and flows between connected nodes.
        x, flow = self.pinn(feature, flow, edge_index)

        return x, flow
