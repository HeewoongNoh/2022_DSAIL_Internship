import yaml
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim

from GMF import GMF
from MLP import MLP

import dataset
import evaluate
import dataframe

# Setting seed
np.random.seed(2023)
torch.manual_seed(2023)
# Configurations
with open('configuration.yaml') as f:
    config = yaml.safe_load(f)

# Device
device = 'gpu' if torch.cuda.is_available() else 'cpu'

# Model: NeuMF
class NeuMF(nn.Module):
    def __init__(self,n_user,n_item, gmf_dim,mlp_dim,num_layers):
        super().__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.num_layers = num_layers
        self.gmf_latent_dim = gmf_dim
        self.mlp_latent_dim = mlp_dim

        #Embedding layers
        self.u_embed_layer_gmf = nn.Embedding(self.n_user,self.gmf_latent_dim)
        self.i_embed_layer_gmf = nn.Embedding(self.n_item,self.gmf_latent_dim)
        self.u_embed_layer_mlp = nn.Embedding(self.n_user, self.mlp_latent_dim * (2**(num_layers-1)))
        self.i_embed_layer_mlp = nn.Embedding(self.n_item, self.mlp_latent_dim * (2**(num_layers-1)))

        #MLP
        mlp_modules = []
        for num_layer in range(num_layers):
            input_dim = self.mlp_latent_dim * (2**(num_layers - num_layer))
            mlp_modules.append(nn.Linear(input_dim, input_dim//2))
            mlp_modules.append(nn.ReLU())
        self.mlp_layers = nn.Sequential(*mlp_modules)

        # GMF
        self.fc_linear_layer = nn.Linear(self.mlp_latent_dim + self.gmf_latent_dim, 1)

    def forward(self, u, i):

        user_embedding_gmf = self.u_embed_layer_gmf(u)
        item_embedding_gmf = self.i_embed_layer_gmf(i)
        user_embedding_mlp = self.u_embed_layer_mlp(u)
        item_embedding_mlp = self.i_embed_layer_mlp(i)

        #Concat embedding vectors together
        interaction_vector = torch.cat((user_embedding_mlp,item_embedding_mlp),-1)
        output_mlp = self.mlp_layers(interaction_vector)

        # Hadmard Product
        element_wise_product = torch.mul(user_embedding_gmf,item_embedding_gmf)

        #Concat gmf + mlp to one vector
        concat_output = torch.cat([element_wise_product, output_mlp], dim=-1)
        final_output_fc = self.fc_linear_layer(concat_output)
        final_output = torch.sigmoid(final_output_fc)

        return final_output

    def load_pretrained_weights(self):

        #Weight from pretrained MLP

        mlp_model = MLP(6040, 3706, config['mlp_dim'], config['num_layers'])
        mlp_model.to(device)
        state_dict = torch.load('checkpoints/MLP_20_Model',map_location=device)
        mlp_model.load_state_dict(state_dict)
        self.u_embed_layer_mlp.weight.data = mlp_model.u_embedding_layer.weight.data
        self.i_embed_layer_mlp.weight.data = mlp_model.i_embedding_layer.weight.data

        for i in range(len(self.mlp_layers),2): # To avoid getting RELU, set step 2
            self.mlp_layers[i].weight.data = mlp_model.mlp_layers[i].weight.data

        # Weight from pretrained GMF
        gmf_model = GMF(6040, 3706, config["gmf_dim"])
        gmf_model.to(device)
        state_dict = torch.load('checkpoints/GMF_20_Model',map_location=device)
        gmf_model.load_state_dict(state_dict)

        self.u_embed_layer_gmf.weight.data = gmf_model.u_embedding_layer.weight.data
        self.i_embed_layer_gmf.weight.data = gmf_model.i_embedding_layer.weight.data

        #To determine trade-off between pretrained models by alpha weight
        self.fc_linear_layer.weight.data = torch.cat(
            [config['alpha']* mlp_model.fc_layer.weight.data, (1-config['alpha'])* gmf_model.linear_layer.weight.data], dim=-1)
        self.fc_linear_layer.bias.data = config['alpha']*mlp_model.fc_layer.bias.data + (1-config['alpha'])*gmf_model.linear_layer.bias.data




#data
ratings =dataframe.ratings
sample_generator = dataset.SampleGenerator(data=ratings)
test_data = sample_generator.evaluate_data


model = NeuMF(6040,3706,config['gmf_dim'],config['mlp_dim'],config['num_layers'])
model.load_pretrained_weights()
model.to(device)
loss_function = nn.BCEWithLogitsLoss()
# optimizer = optim.SGD(model.parameters(),lr=config['learning_rate'])
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
# print(model.parameters())

#Training
record = defaultdict(list)
for epoch in range(1, config['epochs']+1):
    train_loader = sample_generator.instance_a_train_loader(config["num_ng"], config["batch_size"])
    start = time.time()
    model.train()
    total_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        user, item, rating = batch[0], batch[1], batch[2]
        user, item, rating = user.to(device), item.to(device), rating.float().to(device)
        #Update mini-batch
        optimizer.zero_grad()
        prediction = model(user, item)
        loss = loss_function(prediction.view(-1), rating)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    hit_ratio, ndcg = evaluate.cal_metrics(model, test_data)
    end = time.time()
    print(f'Epoch:{epoch} === HR:{hit_ratio:.4f} === NDCG:{ndcg:.4f} === Elapsed time:{end-start:.4f}sec')
    # if epoch == (config['epochs']):
    torch.save(model.state_dict(),f'checkpoints/NeuMF_{epoch}_Model_HR{hit_ratio:.4f}_NCDG{ndcg:.4f}')
        # In the last epoch, model name will be NeuMF_(no of epochs)_Model
    print(f'Model:NeuMF_{epoch}_HR{hit_ratio:.4f}_NDCG:{ndcg:.4f} saving completed')
    record['loss'].append(total_loss)
    record['HR'].append(hit_ratio)
    record['NDCG'].append(ndcg)

fig, axes = plt.subplots(1,2)
axes[0].plot(record['loss'], label='Training loss')
axes[1].plot(record['HR'], label='HR@10')
axes[1].plot(record['NDCG'], label='NDCG@10')
axes[0].legend()
axes[1].legend()
plt.plot()
plt.show()