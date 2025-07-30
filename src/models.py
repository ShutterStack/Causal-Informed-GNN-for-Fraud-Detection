import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn import SAGEConv, global_mean_pool

class GraphSAGEModel(torch.nn.Module):
    """
    A GraphSAGE model for node classification.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # First GraphSAGE layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Second GraphSAGE layer
        x = self.conv2(x, edge_index)
        
        # We return the raw logits for BCEWithLogitsLoss
        return x
