import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Linear


#######################
class Compensation(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels = 256, num_layers = 3):
        super(Compensation, self).__init__()

        # First layer
        self.convs = torch.nn.ModuleList()
        self.convs.append(Linear(in_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):  # Exclude first & last layer
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Output layer
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, z, omega):

        # dropout only for training
        for conv in self.convs[:-1]:  # Hidden layers with ReLU activation
            omega = conv(omega)
            omega = F.relu(omega)  # ReLU activation
            omega = F.dropout(omega, p=0.5)
        
        omega = self.convs[-1](omega)  # Final layer (no activation for classification)

        return z * omega


#######################
class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels = 128, num_layers = 3, dropout = 0.5):
        super(GCN, self).__init__()

        self.dp = dropout
        
        # First layer
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):  # Exclude first & last layer
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Output layer
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index, state):

        # dropout only for training
        if state:
            for conv in self.convs[:-1]:  # Hidden layers with ReLU activation
                x = conv(x, edge_index)
                x = F.relu(x)  # ReLU activation
                x = F.dropout(x, p=self.dp)
        else:
            for conv in self.convs[:-1]:
                x = conv(x, edge_index)
                x = F.relu(x)
        
        x = self.convs[-1](x, edge_index)  # Final layer (no activation for classification)

        return x
    

#######################
class training_GCN(torch.nn.Module):
    def __init__(self, in_c, out_c, hid_c = 128, l_gcn = 3, dp = 0.5):
        super(training_GCN, self).__init__()

        self.compensation = Compensation(in_c, out_c)

        self.gcn = GCN(in_c, out_c, hid_c, l_gcn, dp)

    def forward(self, x, edge_index, omega, partition):

        if self.training:
            # x, edge_index == x', A'
            x = self.gcn(x, edge_index, self.training) # get z'
            x = torch.spmm(partition, x) # P^T * z'
            x = x + self.compensation(x, omega)
        else:
            # x, edge_index == x, A
            x = self.gcn(x, edge_index, self.training)

        return F.log_softmax(x, dim=1)
