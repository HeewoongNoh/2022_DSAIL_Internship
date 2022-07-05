import torch
import torch.nn.functional as F
from torch import nn
import torch.nn.init as weight_init # if you want to initialize weight additionally

#Using pytorch Embeddins
class MF_torch(nn.Module):
    def __init__(self, R, n_users, n_items, n_factors):
        super().__init__()
        self._user_factors = nn.Embedding(n_users,n_factors)
        self._item_factors = nn.Embedding(n_items,n_factors)
        self._R = R
        self._n_factors = n_factors

        #weight initialization

    def forward(self, user, item):
        pred = (self._user_factors(user) * self._item_factors(item)).sum(1)
        return pred

    def full_matrix(self):
        full_matrix = torch.matmul(self._user_factors.weight, self._item_factors.weight.T)
        return full_matrix







