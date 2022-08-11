import math
import torch
from torch import Tensor

# Summary function
def summary(z):
    return torch.sigmoid(z.mean(dim=0))

# Loss function
def loss_function(pos_loss, neg_loss):
    
    epsilon = 1e-15
    pos_loss = -torch.log(pos_loss + epsilon).mean()
    neg_loss = -torch.log(1 -neg_loss + epsilon ).mean()
    
    return pos_loss + neg_loss

# Evaluation for transductive learning
def evaluate(train_z, train_y, test_z, test_y, solver='lbfgs', multi_class='auto', max_iter=150):
    
    #For logReg 
    from sklearn.linear_model import LogisticRegression

    logreg = LogisticRegression(solver=solver, multi_class=multi_class).fit(train_z.detach().cpu().numpy(), train_y.detach().cpu().numpy())
    
    # Mean accuracy                        
    acc = logreg.score(test_z.detach().cpu().numpy(), test_y.detach().cpu().numpy())                           
    return acc

# Corruption function
def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index


# Initialization & Normalization
def reset(value):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)
            
def uniform(size: int, value):
    if isinstance(value, Tensor):
        bound = 1.0 / math.sqrt(size)
        value.data.uniform_(-bound, bound)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            uniform(size, v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            uniform(size, v)
