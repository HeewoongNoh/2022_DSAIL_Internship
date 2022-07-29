import yaml
import time
import numpy as np
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from model import GNNStack


# Setting seed
torch.manual_seed(2023)

# Device
device = 'gpu' if torch.cuda.is_available() else 'cpu'

# Configurations
with open('configuration.yaml') as f:
    config = yaml.safe_load(f)


def train(dataset):
    test_set_size = np.sum(dataset[0]['train_mask'].numpy())
    print(f'Node task...test_set_size: {test_set_size}')
    test_loader = loader = DataLoader(dataset, config['batch_size'], shuffle=True)

    # build model
    model = GNNStack(dataset.num_node_features, config['hidden_dim'], dataset.num_classes,config['dropout'],num_layers=config['num_layers'])
    optimizer = optim.Adam(model.parameters(),lr=config['lr'], weight_decay=config['weight_decay'])

    # train
    losses = []
    test_accs = []

    for epoch in range(config['epochs']):
        start = time.time()
        total_loss = 0
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            pred = model(batch)
            label = batch.y
            pred = pred[batch.train_mask]
            label = label[batch.train_mask]
            loss = model.loss(pred, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(loader.dataset)
        losses.append(total_loss)
        end = time.time()
        print(f"EPOCH:{epoch+1}===Training_loss:{total_loss:.4f}===Elapsed_time:{end-start:.4f}")
        if epoch % 10 == 0:
            test_acc = test(test_loader, model)
            test_accs.append(test_acc)
        else:
            test_accs.append(test_accs[-1])
    return test_accs, losses


def test(loader, model, is_validation=True):
    model.eval()

    correct = 0
    for data in loader:
        with torch.no_grad():
            # max(dim=1) returns values, indices tuple; only need indices
            pred = model(data).max(dim=1)[1]

        mask = data.val_mask if is_validation else data.test_mask

        # node classification: only evaluate on nodes in test set
        pred = pred[mask]
        label = data.y[mask]
        correct += pred.eq(label).sum().item()

    total = 0
    for data in loader.dataset:
        total += torch.sum(data.val_mask if is_validation else data.test_mask).item()
    return correct / total
