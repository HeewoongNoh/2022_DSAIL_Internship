# Import library
import yaml
import time
import torch
from model import Transductive_DGI, Encoder
from utils import corruption, loss_function, evaluate, summary
from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt
# Setting seed
torch.manual_seed(2023)

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Configurations
with open('C:/Users/heewo/Desktop/vscode_project/2022_DSAIL_Internship/DGI_pyg/configuration.yaml') as f:
    config = yaml.safe_load(f)

# Dataset
dataset = Planetoid(root='/tmp/cora', name='Cora')
data = dataset[0].to(device)

# Model
model = Transductive_DGI(config['hidden_dim'], encoder=Encoder(dataset.num_features, config['hidden_dim']), summary=summary, corruption=corruption)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])


def train():
    training_loss = []
    for epoch in range(config['epochs']):
        start = time.time()
        total_loss = 0.0
        model.train()
        optimizer.zero_grad()
        pos_z, neg_z, summary ,pos_loss, neg_loss = model(data)
        loss = loss_function(pos_loss, neg_loss)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        training_loss.append(total_loss)
        end = time.time()
        print(f'EPOCH:{epoch+1}===Training_loss:{total_loss:.4f}===Elapsed_time:{end-start:.4f}')
    plt.title(dataset.name)
    plt.plot(training_loss, label="Transductive_DGI_training_loss")
    plt.legend()
    plt.show()
    
def test():
    model.eval()
    z, _, _, _, _ = model(data)
    acc = evaluate(z[data.train_mask], data.y[data.train_mask], z[data.test_mask], data.y[data.test_mask], max_iter=150)
    print(f'Dataset: {dataset.name} >>> DGI_Transductive_classification_acc: {acc:.4f}')



if __name__ == '__main__':
    train()
    test()