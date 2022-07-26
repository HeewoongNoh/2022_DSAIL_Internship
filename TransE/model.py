import torch
import numpy as np
import torch.nn as nn

np.random.seed(2022)
torch.manual_seed(2022)

# Model TransE
class transE(nn.Module):
    def __init__(self, e_count, r_count, device, norm, dim, margin):
        super(transE, self).__init__()
        self.e_count = e_count
        self.r_count = r_count
        self.device = device
        self.norm = norm
        self.dim = dim
        self.margin = margin
        self.criterion = nn.MarginRankingLoss(margin=self.margin, reduction='none')
        self.e_embedding = self.init_entity()
        self.r_embedding = self.init_relation()

    def init_relation(self):
        r_embedding = nn.Embedding(num_embeddings=self.r_count + 1, embedding_dim=self.dim, padding_idx=self.r_count)
        uniform_normalize = 6 / np.sqrt(self.dim)
        r_embedding.weight.data.uniform_(uniform_normalize, uniform_normalize)
        #Normalize
        # r_norm = torch.norm(r_embedding.weight.data[:-1, :], p=1, dim=1, keepdim=True)
        # r_embedding.weight.data = r_embedding.weight.data[:-1, :] / r_norm
        r_embedding.weight.data[:-1, :].div_(r_embedding.weight.data[:-1,:].norm(p=1, dim=1, keepdim=True))
        return r_embedding


    def init_entity(self):
        e_embedding = nn.Embedding(num_embeddings=self.e_count + 1, embedding_dim=self.dim, padding_idx=self.e_count)
        uniform_normalize = 6 / np.sqrt(self.dim)
        e_embedding.weight.data.uniform_(uniform_normalize, uniform_normalize)
        return e_embedding

    def forward(self, pos_triplets, neg_triplets):
        # Normalize entity in the loop section, not in initialization section
        # e_norm = torch.norm(self.e_embedding.weight.data[:-1, :], p=1, dim=1, keepdim=True)
        # self.e_embedding.weight.data = self.e_embedding.weight.data[:-1, :] / e_norm
        self.e_embedding.weight.data[:-1, :].div_(self.e_embedding.weight.data[:-1, :].norm(p=1, dim=1, keepdim=True))
        if pos_triplets.size()[1] == 3:
            pos_distance = self.distance(pos_triplets)
        if neg_triplets.size()[1] == 3:
            neg_distance = self.distance(neg_triplets)
        else:
            raise AssertionError

        return self.loss(pos_distance, neg_distance), pos_distance, neg_distance

    #calculate dissimilarity score for triplets
    def predict(self, triplets):
        return self.distance(triplets)

    def loss(self, pos_distance, neg_distance):
        # For using MarginRankingLoss, target should be '-1'
        target = torch.tensor([-1], dtype=torch.long, device=self.device)
        return self.criterion(pos_distance, neg_distance, target)

    def distance(self, triplets):
        if triplets.size()[1] == 3:
            head = triplets[:, 0]
            relation = triplets[:, 1]
            tail = triplets[:, 2]
        else:
            raise AssertionError
        return (self.e_embedding(head) + self.r_embedding(relation) - self.e_embedding(tail)).norm(p=self.norm, dim=1)














