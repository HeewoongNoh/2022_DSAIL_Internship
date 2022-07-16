import numpy as np
import matplotlib.pyplot as plt
import yaml
from collections import defaultdict
from dataset import trainset, testset, movie_feature, preprocess_dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

 # Model using 'ml-latest-small' as dataset

#Setting Seed
np.random.seed(2023)
torch.manual_seed(2023)

#Configurations
with open('configuration.yaml') as f:
    config = yaml.safe_load(f)

#Dataset from dataset.py
train = trainset
test = testset
item = movie_feature
print(train.shape, test.shape, item.shape)

#Negative item list and positive item
train_neg_item, test_neg_item, train_pos_item_bool, test_pos_item_bool =preprocess_dataset(train,test)


class CMLData(Dataset):
    def __init__(self, train, item_data, neg_sample_size, neg_item):
        self.trainset = train
        self.item_data = item_data
        self.user_pos = trainset.nonzero()[0]
        self.item_pos = trainset.nonzero()[1]
        self.n_user, self.n_item = trainset.shape
        self.neg_sample_size = neg_sample_size
        self.neg_item_per_user = neg_item

    def __len__(self):
        return len(self.user_pos)

    def __getitem__(self, idx):
        user = self.user_pos[idx]
        item = self.item_pos[idx]
        neg_item = np.random.choice(self.neg_item_per_user[user], self.neg_sample_size)
        item_feature = self.item_data[item, :]
        return {'user_idx':user,'item_idx':item,'neg_item_idx':neg_item,'item_x':item_feature}

class CML(nn.Module):
    def __init__(self, user_n, item_n, input_dim, latent_dim, neg_sample_size, margin, neg_item):
        super().__init__()
        self.user_n = user_n
        self.item_n = item_n
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.neg_sample_size = neg_sample_size
        self.margin = margin
        self.neg_item = neg_item
        self.lambda_f = config['lambda_f'] #Hyperparamter: Constant for feature loss,loss_f
        self.lambda_c = config['lambda_c'] #Hyperparamter: Constant for covariance loss,loss_c
        self.embedding_user = nn.Embedding(user_n, latent_dim, max_norm=1) #Max norm: 1
        self.embedding_item = nn.Embedding(item_n,latent_dim, max_norm=1)
        self.dropout = nn.Dropout(p=0.5)

        # Mlp with 256 dimensional hidden layer
        mlp_modules = []
        mlp_modules.append(nn.Linear(self.input_dim, 256))
        mlp_modules.append(nn.ReLU())
        mlp_modules.append(self.dropout)
        mlp_modules.append(nn.Linear(256, self.latent_dim))
        mlp_modules.append(nn.ReLU())
        mlp_modules.append(self.dropout)
        self.mlp_layers = nn.Sequential(*mlp_modules)
        self.fc_layer = nn.Linear(self.latent_dim, 1)

        #Initialize rank_d_ij
        self.rank_d_ij = torch.normal(mean=self.item_n/2, std= 1,size=(user_n,item_n))

    def forward(self, batch_data):
        user, item, neg_item, item_x = batch_data['user_idx'], batch_data['item_idx'], batch_data['neg_item_idx'], \
                                       batch_data['item_x'].float()
        batch_size = user.size(0)
        w_ij = self.rank_d_ij[user, item].unsqueeze(-1)
        ui, vj, vk = self.embedding_user(user), self.embedding_item(item), self.embedding_item(neg_item)

        # loss margin
        pos_distance = self.pos_distance(ui, vj)
        neg_distance = self.neg_distance(ui, vk)
        hinged_loss = torch.relu(pos_distance - neg_distance + self.margin)
        loss_m = torch.sum(w_ij * hinged_loss, axis=1)

        # loss feature
        item_x = self.item_feature_extractor(item_x)
        loss_f = torch.sum((item_x - vj) ** 2, axis=1)

        # loss covariance (covariance regularization)
        C = self.covariance_matrix(ui, vj, batch_size)
        loss_c = (torch.norm(C, p='fro') - torch.norm(torch.diagonal(C, 0), 2)) / batch_size

        self.update_rank(pos_distance, neg_distance, user, item)

        return torch.sum(loss_m), torch.sum(loss_f), loss_c, torch.sum(loss_m) + torch.sum(
            loss_f) * self.lambda_f + loss_c * self.lambda_c

    def update_rank(self, pos_d, neg_d, user, item):
        impostor_M = torch.sum((pos_d + self.margin - neg_d) > 0, axis=1)
        self.rank_d_ij[user, item] = torch.log((impostor_M* self.item_n / self.neg_sample_size) + 1)

    def pos_distance(self, ui, vj):
        return torch.sum((ui - vj) ** 2, axis=1).unsqueeze(-1)

    def neg_distance(self, ui, vk):
        return torch.sum((ui.unsqueeze(axis=1) - vk) ** 2, axis=2)

    def item_feature_extractor(self, item_feature):
        item_feature = self.mlp_layers(item_feature)
        item_feature = self.fc_layer(item_feature)
        return item_feature

    def covariance_matrix(self, ui, vj, batch_size):
        cat_embedding = torch.cat((ui, vj), axis=0)
        mu = torch.mean(cat_embedding, axis=0)
        cat_embedding = cat_embedding - mu
        C = torch.matmul(cat_embedding.T, cat_embedding) / batch_size
        return C


def eval_recallM(model):
    recall_tr = []
    recall_tst = []

    for u in torch.arange(train.shape[0]).to(config['device']):
        ui = model.embedding_user(u)
        scores = torch.sum((ui - model.embedding_item.weight.data) ** 2, axis=1).detach().cpu().numpy()
        rank = scores.argsort().argsort()
        topM_mask = (rank <= config['rankM'])
        pos_item_mask_tr = train_pos_item_bool[u.item()]
        pos_item_mask_tst = test_pos_item_bool[u.item()]
        if pos_item_mask_tr.sum() > 0:
            recall_tr.append((topM_mask * pos_item_mask_tr).sum() / pos_item_mask_tr.sum())
        if pos_item_mask_tst.sum() >= 5:
            recall_tst.append((topM_mask * pos_item_mask_tst).sum() / pos_item_mask_tst.sum())
    return np.mean(recall_tr), np.mean(recall_tst)

if __name__ == '__main__':

    user_pos = train.nonzero()[0]
    item_pos = train.nonzero()[1]
    train_dataset = CMLData(train, item, config['neg_sample_size'], train_neg_item)
    valid_dataset = CMLData(test, item, config['neg_sample_size'], test_neg_item)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], drop_last=False, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'] * 10, drop_last=False, shuffle=False)

    model = CML(train.shape[0],
                train.shape[1],
                item.shape[1],
                config['latent_dim'],
                config['neg_sample_size'],
                config['margin'],
                train_neg_item,
                )

    model = model.to(config['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay= config['weight_decay'])

    history = defaultdict(list)
    for epoch in range(config['epochs']):
        losses = []
        model.train()
        for batch_data in train_loader:
            optimizer.zero_grad()
            batch_data = {k: v.to(config['device']) for k, v in batch_data.items()}
            loss_m, loss_f, loss_c, loss = model(batch_data)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        recall_tr, recall_tst = eval_recallM(model)
        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == config['epochs']:
            config_rank = config['rankM']
            print(f'EPOCH {epoch + 1} : Training loss {np.sum(losses) : .0f}, trainset recall@{config_rank} {recall_tr: .4f}, validation recall@{config_rank} {recall_tst: .4f}')
        history['loss'].append(np.sum(losses))
        history['recall_tr'].append(recall_tr)
        history['recall_tst'].append(recall_tst)

    fig, axes = plt.subplots(1,2)
    axes[0].plot(history['loss'], label='Training loss')
    axes[1].plot(history['recall_tr'], label='Training-recall@50')
    axes[1].plot(history['recall_tst'], label='Validaton-recall@50')
    axes[0].legend()
    axes[1].legend()
    plt.plot()
    plt.show()


# Reference
# (Official) https://github.com/changun/CollMetric/blob/master/CML.py
# https://github.com/yeonjun-in/GNN_Recsys_paper/blob/main/rec/CML/main.py"