# Import library
import yaml
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
# from torch.utils.tensorboard import SummaryWriter


# Setting Seed
torch.manual_seed(2023)

# Device
device = 'gpu' if torch.cuda.is_available() else 'cpu'

# Configurations
with open('C:/Users/heewo/Desktop/vscode_project/2022_DSAIL_Internship/VAE/configuration.yaml') as f:
    config = yaml.safe_load(f)

# Model 
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim,num_layers,latent_dim):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=0.2)

        #Encoder part
        self.vae_e = nn.ModuleList()
        self.vae_e.append(nn.Linear(input_dim,hidden_dim))                    # (784, 512)
        
        if num_layers >=1:                                  #if num_layer ==2: (512, 256)  
            for _ in range(num_layers):                                       #(256, 128)
                half_dim = hidden_dim//2
                self.vae_e.append(nn.Linear(hidden_dim, half_dim))
                hidden_dim = half_dim
        
        #output of encoder
        self.vae_mu = nn.Linear(half_dim, latent_dim)                         # (128, 64)
        self.vae_sig = nn.Linear(half_dim, latent_dim)                        # (128, 64)
        
        #Decoder part (vice versa of enocder part)
        self.vae_d = nn.ModuleList()
        self.vae_d.append(nn.Linear(latent_dim,half_dim))                     # (64, 128)
        
        if num_layers >=1:                                 # Opposite structure (128, 256)
            for _ in range(num_layers):                                        #(256, 512)
                double_dim = 2 * half_dim
                self.vae_d.append(nn.Linear(half_dim,double_dim))
                half_dim = double_dim
                
        self.vae_fc = nn.Linear(double_dim, input_dim)                         # (512,784)
        

            
    def forward(self,x):
        x = x.reshape(-1,784)         #flattening for linear layer
        # encoder
        for i in range(self.num_layers+1):
            x = self.vae_e[i](x)
            x = F.relu(x)
            x = self.dropout(x)
            
        mu = self.vae_mu(x)
        sig = self.vae_sig(x)
        std = torch.exp(0.5 * sig)
        eps = torch.randn_like(std)
        
        output = mu + std * eps
        
        # decoder
        for i in range(self.num_layers+1):
            output = self.vae_d[i](output)
            output = F.relu(output)
            output = self.dropout(output)
            
        output = self.vae_fc(output)
        sigmoid_output = torch.sigmoid(output)
        
        return sigmoid_output, mu, sig
        
        
        
  