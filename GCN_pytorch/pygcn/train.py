from __future__ import division
from __future__ import print_function

import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy
from pygcn.models import GCN_layer_3, GCN
#Tensorboard
from torch.utils.tensorboard import SummaryWriter

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=400,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# set the seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer
# Original GCN
'''
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
# 3 layer GCN
'''
model = GCN_layer_3(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)


# If you want to use CUDA (gpu)
if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

#Modified part from original code
def train_mod():
    # Use tensorboard
    # log_dir =
    writer = SummaryWriter('runs/GCN_layer_2_experiment_1')
    for epoch in range(args.epochs):

        print(f'now training in {epoch+1} epoch')
        output = model(features,adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        model.eval() # deactivate dropout layer for inference

        output = model(features,adj)
        acc_train = accuracy(output[idx_train], labels[idx_train])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        model.train() # switch to training mode

        #Tensorboard add_scalars
        writer.add_scalars('loss',{'loss_train':loss_train.detach().numpy(),'loss_val':loss_val.detach().numpy()},epoch)
        writer.add_scalars('accuracy',{'acc_train':acc_train,'acc_val':acc_val},epoch)

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

    writer.close()
    print(f'!!!finished!!!')

if __name__ == '__main__':
    train_mod()

