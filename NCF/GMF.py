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


#Model: GMF
class GMF(nn.Module):
    def __init__(self, n_user,n_item,gmf_dim):
        super().__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.latent_dim = gmf_dim

        #Embedding layers
        self.u_embedding_layer = nn.Embedding(self.n_user,self.latent_dim)
        self.i_embedding_layer = nn.Embedding(self.n_item,self.latent_dim)
        #Nomrmalizng?

        #Linear layer
        self.linear_layer = nn.Linear(self.latent_dim,1)

    def forward(self,u,i):
        user_embedding = self.u_embedding_layer(u)
        item_embedding = self.i_embedding_layer(i)
        element_wise_product = torch.mul(user_embedding,item_embedding) #Hadmard Product
        mid_output = self.linear_layer(element_wise_product)
        output = torch.sigmoid(mid_output)
        return output


if __name__ == '__main__':

    #Data
    ratings =dataframe.ratings
    sample_generator = dataset.SampleGenerator(data=ratings)
    test_data = sample_generator.evaluate_data


    model= GMF(6040,3706,config["gmf_dim"]).to(device)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(),lr=config['learning_rate'])
    # writer = SummaryWriter(logdir='runs/GMF')
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
        # torch.save(model.state_dict(),f'checkpoints/GMF_{epoch}_HR{hit_ratio:.4f}_NDCG_{ndcg:.4f}')
        torch.save(model.state_dict(),f'checkpoints/GMF_{epoch}_Model')
        print(f'Model:GMF_{epoch}_HR{hit_ratio:.4f}_NDCG:{ndcg:.4f} saving completed')

