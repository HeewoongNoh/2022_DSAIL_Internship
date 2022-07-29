'''
Practice using pyG for implementing graphSAGE Model
'''

# Import library
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing


class GNNStack(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,dropout,num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GraphSage(input_dim, hidden_dim,bias=True))

        if num_layers >=1:
            for i in range(num_layers -1):
                self.convs.append(GraphSage(hidden_dim, hidden_dim))
        self.post_mp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim,output_dim))

        self.dropout = dropout
        self.num_layers = num_layers

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i in range(self.num_layers):
            x = self.convs[i](x,edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)
        x = self.post_mp(x)
        return F.log_softmax(x,dim=1)

    def loss(self, pred,label):
        return F.nll_loss(pred,label)

class GraphSage(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True):
        super().__init__(aggr='mean')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        self.lin = nn.Linear(self.in_channels, self.out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index, size=None):

        # propagate() internally calls message(), aggregate(), update()
        x = self.lin(x)

        # default normalize l2
        if self.normalize:
            x = F.normalize(x)

        out = self.propagate(edge_index, x=x)
        x += self.bias

        return out

    def message(self, x_j):
        return x_j







