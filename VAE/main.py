# Import library
import yaml
import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from model import VAE
from train import train, test



#Setting Seed
torch.manual_seed(2023)

# Device
device = 'gpu' if torch.cuda.is_available() else 'cpu'

#Configurations
with open('C:/Users/heewo/Desktop/vscode_project/2022_DSAIL_Internship/VAE/configuration.yaml') as f:
    config = yaml.safe_load(f)
# Loading Dataset and
# Normalized to [0,1] by ToTensor()
trainset = torchvision.datasets.MNIST(root = 'C:/Users/heewo/Desktop/vscode_proejct/2022_DSAIL_intership/VAE/MNIST', train = True,
                                        download = True, transform = transforms.ToTensor())

trainloader = torch.utils.data.DataLoader(trainset, batch_size = config['batch_size'], shuffle = True)


testset = torchvision.datasets.MNIST(root = 'C:/Users/heewo/Desktop/vscode_proejct/2022_DSAIL_intership/VAE/MNIST', train = False,
                                        download = True, transform = transforms.ToTensor())

testloader = torch.utils.data.DataLoader(testset, batch_size = config['batch_size'], shuffle = True)



def main():

    train(trainloader)
    vae_model = VAE(784,config['hidden_dim'],config['num_layers'],config['latent_dim']).to(device)
    state_dict_vae = torch.load('C:/Users/heewo/Desktop/vscode_project/2022_DSAIL_Internship/VAE/checkpoints/VAE_model')
    vae_model.load_state_dict(state_dict_vae)
    test(vae_model,testloader)
    
if __name__ == '__main__':
    main()

