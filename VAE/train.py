# Import library
import yaml
import time
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
from model import VAE
# from torch.utils.tensorboard import SummaryWriter


#Setting Seed
torch.manual_seed(2023)

# Device
device = 'gpu' if torch.cuda.is_available() else 'cpu'

#Configurations
with open('C:/Users/heewo/Desktop/vscode_project/2022_DSAIL_Internship/VAE/configuration.yaml') as f:
    config = yaml.safe_load(f)


def loss_function(input_x, x, mu, sig):
    BCE = F.binary_cross_entropy(input_x, x.view(-1, 784), reduction = 'sum')
    KLD = -0.5 * torch.sum(1 + sig - mu.pow(2) - sig.exp())
    return BCE, KLD



def train(train_loader):
    
    # loading model
    model = VAE(784,config['hidden_dim'],config['num_layers'],config['latent_dim'])
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    
    # train
    losses = []
    
    for epoch in range(config['epochs']):
        start = time.time()
        train_loss = 0
        model.train()
        for (data, _) in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, sig = model(data)
            BCE, KLD = loss_function(recon_batch, data, mu, sig)
            loss = BCE + KLD
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        train_loss /= len(train_loader.dataset)
        losses.append(train_loss)
        end = time.time()
        print(f'EPOCH:{epoch+1}===Training loss:{train_loss:.4f}===Elapsed_time:{end-start:.4f}')
    torch.save(model.state_dict(),f'C:/Users/heewo/Desktop/vscode_project/2022_DSAIL_Internship/VAE/checkpoints/VAE_model_1')
    plt.title('training_loss')
    plt.plot(losses, label="VAE_training_loss")
    plt.legend()
    plt.show()
    

def test(model, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_loader):
            data = data.to(device)
            
            recon_batch, mu, sig = model(data)
            BCE, KLD = loss_function(recon_batch, data, mu, sig)

            loss = BCE + KLD
            test_loss += loss.item()

            if batch_idx == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch.view(config['batch_size'], 1, 28, 28)[:n]]) # (16, 1, 28, 28)
                grid = torchvision.utils.make_grid(comparison.cpu()) # (3, 62, 242)
                plt.imshow(grid.permute(1,2,0))
                plt.show()
                
                
                


