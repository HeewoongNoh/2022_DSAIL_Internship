# Import library
import yaml
import torch
from train import train, test
from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt

# Setting seed
torch.manual_seed(2023)

# Device
device = 'gpu' if torch.cuda.is_available() else 'cpu'

# Configurations
with open('configuration.yaml') as f:
    config = yaml.safe_load(f)

def main():

    dataset = Planetoid(root='/tmp/cora', name='Cora')
    test_accs, losses = train(dataset)

    print(f'Max accuracy:{max(test_accs)}')
    print(f'Min loss:{min(losses)}')

    plt.title(dataset.name)
    plt.plot(losses, label="GraphSAGE_training_loss")
    plt.plot(test_accs, label="GraphSAGE_test_accuracy")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()