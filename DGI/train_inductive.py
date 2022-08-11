import time
import yaml
import torch
import torch.nn as nn
from model import Inductive_DGI
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv
from utils import corruption, summary, evaluate, loss_function
import matplotlib.pyplot as plt
# Setting seed
torch.manual_seed(2023)

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Configurations
with open('C:/Users/heewo/Desktop/vscode_project/2022_DSAIL_Internship/DGI_pyg/configuration.yaml') as f:
    config = yaml.safe_load(f)

dataset = Reddit(root='/tmp/data/Reddit')
data = dataset[0].to(device)

train_loader = NeighborSampler(data.edge_index, node_idx=None,
                               sizes=[10, 10, 25], batch_size=256,
                               shuffle=True)

test_loader = NeighborSampler(data.edge_index, node_idx=None,
                              sizes=[10, 10, 25], batch_size=256,
                              shuffle=False)


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            SAGEConv(in_channels, hidden_channels),
            SAGEConv(hidden_channels, hidden_channels),
            SAGEConv(hidden_channels, hidden_channels)
        ])

        self.activations = torch.nn.ModuleList()
        self.activations.extend([
            nn.PReLU(hidden_channels),
            nn.PReLU(hidden_channels),
            nn.PReLU(hidden_channels)
        ])

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            x = self.activations[i](x)
        return x

model = Inductive_DGI(config['hidden_dim'], encoder=Encoder(dataset.num_features, config['hidden_dim']), summary=summary, corruption=corruption)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

x, y = data.x.to(device), data.y.to(device)


def train(epoch):
    model.train()

    total_loss = total_examples = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()
        pos_z, neg_z, summary, pos_loss, neg_loss = model(x[n_id], adjs)
        loss = loss_function(pos_loss,neg_loss)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pos_z.size(0)
        total_examples += pos_z.size(0)

    return total_loss / total_examples


@torch.no_grad()
def test():
    model.eval()

    zs = []
    for i, (batch_size, n_id, adjs) in enumerate(test_loader):
        adjs = [adj.to(device) for adj in adjs]
        zs.append(model(x[n_id], adjs)[0])
        
    z = torch.cat(zs, dim=0)
    train_val_mask = data.train_mask | data.val_mask
    acc = evaluate(z[train_val_mask], y[train_val_mask], z[data.test_mask],
                     y[data.test_mask], max_iter=10000)
    return acc

training_loss = []
for epoch in range(1, 16):
    loss = train(epoch)
    training_loss.append(loss)
    print(f'Epoch {epoch}, Loss: {loss:.4f}')
    
    
test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}') 
plt.title(f'Reddit')
plt.plot(training_loss, label="Inductive_DGI_training_loss")
plt.legend()
plt.show()
