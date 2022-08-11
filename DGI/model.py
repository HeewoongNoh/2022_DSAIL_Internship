import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from utils import reset, uniform


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=True) # For transductive , set the true
        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        return x
    
    
class Transductive_DGI(nn.Module):
    def __init__(self, hidden_dim, encoder, summary, corruption):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = encoder
        self.summary = summary
        self.corruption = corruption
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()
        
        
    def reset_parameters(self):
        reset(self.encoder)
        reset(self.summary)
        uniform(self.hidden_dim, self.weight)
        
        
    def forward(self, data):
        
        """Return the latent space for the input args, their corruptions, summary representations"""
        
        x, edge_index = data.x, data.edge_index
        
        pos_z = self.encoder(x, edge_index)
        cor = self.corruption(x, edge_index)
        
        cor = cor if isinstance(cor, tuple) else (cor,)
        
        neg_z = self.encoder(*cor)
        summary = self.summary(pos_z)
        pos_loss = self.discriminate(pos_z, summary, sigmoid=True)
        neg_loss = self.discriminate(neg_z, summary, sigmoid = True)
        
        return pos_z, neg_z, summary, pos_loss, neg_loss
    
    
    def discriminate(self, z, summary, sigmoid=True):
        
        summary = summary.t() if summary.dim() > 1 else summary
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        
        if sigmoid:
            return torch.sigmoid(value)
        else:
            return value
        
        
        
class Inductive_DGI(nn.Module):
    def __init__(self, hidden_dim, encoder, summary, corruption):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = encoder
        self.summary = summary
        self.corruption = corruption
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()
        
        
    def reset_parameters(self):
        reset(self.encoder)
        reset(self.summary)
        uniform(self.hidden_dim, self.weight)
        
        
    def forward(self, x, adjs):
        
        """Return the latent space for the input args, their corruptions, summary representations"""
        
        pos_z = self.encoder(x, adjs)
        cor = self.corruption(x, adjs)
        
        cor = cor if isinstance(cor, tuple) else (cor,)
        
        neg_z = self.encoder(*cor)
        summary = self.summary(pos_z)
        pos_loss = self.discriminate(pos_z, summary, sigmoid=True)
        neg_loss = self.discriminate(neg_z, summary, sigmoid = True)
        
        return pos_z, neg_z, summary, pos_loss, neg_loss
    
    def discriminate(self, z, summary, sigmoid=True):
        
        summary = summary.t() if summary.dim() > 1 else summary
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        
        if sigmoid:
            return torch.sigmoid(value)
        else:
            return value