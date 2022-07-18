import yaml
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import dataset
import evaluate
import dataframe

#Setting seed
np.random.seed(2023)
torch.manual_seed(2023)

#Configurations
with open('configuration.yaml') as f:
    config = yaml.safe_load(f)

#Device
device = 'gpu' if torch.cuda.is_available() else 'cpu'


#Model: MLP
class MLP(nn.Module):
    def __init__(self, n_user, n_item, mlp_dim,num_layers):
        super().__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.latent_dim = mlp_dim
        self.num_layers = num_layers


        #Embedding layers
        self.u_embedding_layer = nn.Embedding(self.n_user, self.latent_dim * (2**(num_layers-1)))
        self.i_embedding_layer = nn.Embedding(self.n_item, self.latent_dim * (2**(num_layers-1)))

        mlp_modules = []
        for num_layer in range(num_layers):
            input_dim = self.latent_dim * (2**(num_layers - num_layer))
            mlp_modules.append(nn.Linear(input_dim, input_dim//2))
            mlp_modules.append(nn.ReLU())
        self.mlp_layers = nn.Sequential(*mlp_modules)
        self.fc_layer = nn.Linear(self.latent_dim,1)

        #Normalizing?


    def forward(self,u,i):
        user_embedding = self.u_embedding_layer(u)
        item_embedding = self.i_embedding_layer(i)

        #Concat embedding vectors together
        interaction_vector = torch.cat((user_embedding,item_embedding),-1)
        output_mlp = self.mlp_layers(interaction_vector)
        output_fc = self.fc_layer(output_mlp)
        output = torch.sigmoid(output_fc)
        return output

if __name__ == '__main__':
    #data
    ratings =dataframe.ratings
    sample_generator = dataset.SampleGenerator(data=ratings)
    test_data = sample_generator.evaluate_data



    model=MLP(6040,3706, config['mlp_dim'],config['num_layers'])
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(),lr=config['learning_rate'])


    #Training
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
        # torch.save(model.state_dict(),f'checkpoints/MLP_{epoch}_HR{hit_ratio:.4f}_NDCG_{ndcg:.4f}')
        torch.save(model.state_dict(), f'checkpoints/MLP_{epoch}_Model')
        print(f'Model:MLP_{epoch}_HR{hit_ratio:.4f}_NDCG:{ndcg:.4f} saving completed')


