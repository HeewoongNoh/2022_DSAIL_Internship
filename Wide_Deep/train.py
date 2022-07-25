#Import library
import yaml
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import WD_Dataset, Wide_Dataset, Deep_Dataset, df_wide_pro,df_deep,df_wide
from model import Wide_Deep, Wide, Deep
from collections import defaultdict
from sklearn.metrics import roc_auc_score, roc_curve

# Setting seed
torch.manual_seed(2023)

# Configurations
with open('configuration.yaml') as f:
    config = yaml.safe_load(f)

# Device
device = 'gpu' if torch.cuda.is_available() else 'cpu'

Y = np.array(df_deep['label'])
df_wide_pro.drop(['label'],axis=1,inplace=True)
df_deep.drop(['label'],axis=1,inplace=True)

# In Original dataset, the data type (train, test) was fixed, so I don't consider using train_test_split
length = 46361 # len(train_df)

#If you want to use df_wide_final.pkl for training wide model
# X_train_wide = df_wide.values[:len(train_df),:]
# X_test_wide= df_wide.values[len(train_df):,:]

#If you want to use df_wide_promotionID.pkl for training wide model
X_train_wide = df_wide_pro.values[:length,:]
X_test_wide= df_wide_pro.values[length:,:]

# Dataset for deep
X_train_deep = df_deep.values[:length,:]
X_test_deep= df_deep.values[length:,:]

# Answer dataset
Y_train = Y[:length]
Y_test = Y[length:]

#Fixed label tensor (Target)
test_wide_tensor = torch.FloatTensor(X_test_wide)
test_deep_tensor = torch.LongTensor(X_test_deep)
test_tensor = torch.FloatTensor(Y_test)


########################### " Wide & Deep Model " #################################

train_dataset = WD_Dataset(X_wide_tensor = torch.FloatTensor(X_train_wide),
                           X_deep_tensor = torch.LongTensor(X_train_deep),Y_tensor = torch.FloatTensor(Y_train))
train_loader = DataLoader(dataset = train_dataset, batch_size= config['batch_size'], shuffle=True)
model = Wide_Deep(df_wide_pro,df_deep).to(device)
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], betas=(0.9, 0.999),
                       eps=1e-08, weight_decay=config['weight_decay'])
criterion = nn.BCELoss()
# criterion = nn.BCEWithLogitsLoss()


auc_wide_deep = []
for epoch in range(config['epochs']):
    start = time.time()
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        X_wide, X_deep, Y = batch[0], batch[1], batch[2]
        X_wide, X_deep, Y = X_wide.to(device), X_deep.to(device), Y.to(device)
        optimizer.zero_grad()
        Y_pred = model(X_wide, X_deep)
        loss = criterion(Y_pred.squeeze(), Y)
        # If you want to use L1 reg instead of l2 reg.
        # Set the adam's weight decay:0
        # Compute L1 loss
        # l1_weight =0.01
        # l1_parameters = []
        # for parameter in model.parameters():
        #     l1_parameters.append(parameter.reshape(-1))
        # l1 = l1_weight * model.compute_L1_loss(torch.cat(l1_parameters))
        # loss += l1
        loss.backward()
        optimizer.step()
        total_loss += loss

    model.eval()
    pred = model(test_wide_tensor.to(device), test_deep_tensor.to(device))
    auc = roc_auc_score(test_tensor.cpu().detach().numpy(), pred.cpu().detach().numpy())
    auc_wide_deep.append(auc)
    end = time.time()
    if epoch >= 0:
        print(f'EPOCH:{epoch + 1} === loss:{total_loss:.4f} === AUC:{auc:.4f} ===Elapsed:{(end - start):.4f}')
        torch.save(model.state_dict(),f'checkpoints/Wide_Deep_Model_Epoch_{epoch+1}_AUC_{auc:.4f}')
print(f'Wide_Deep_Model_Best_AUC:{sorted(auc_wide_deep)[-1]:.4f}')
########################### " Wide Model " #################################
# For Wide
model_name = 'Wide'
train_dataset_wide = Wide_Dataset(X_wide_tensor=torch.FloatTensor(X_train_wide), Y_tensor=torch.FloatTensor(Y_train))
train_loader_wide = DataLoader(dataset=train_dataset_wide, batch_size=config['batch_size'], shuffle=True)
model = Wide(df_wide_pro).to(device)

optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], betas=(0.9, 0.999), eps=1e-08, weight_decay=config['weight_decay'])
criterion = nn.BCELoss()
# criterion = nn.BCEWithLogitsLoss()

auc_wide = []

for epoch in range(config['epochs']):
    start = time.time()
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(train_loader_wide):
        X_wide, Y = batch[0], batch[1]
        X_wide, Y = X_wide.to(device), Y.to(device)
        optimizer.zero_grad()
        Y_pred = model(X_wide)
        loss = criterion(Y_pred.squeeze(), Y)

        # If you want to use L1 reg instead of l2 reg.
        # Set the adam's weight decay:0
        # Compute L1 loss
        # l1_weight =0.01
        # l1_parameters = []
        # for parameter in model.parameters():
        #     l1_parameters.append(parameter.reshape(-1))
        # l1 = l1_weight * model.compute_L1_loss(torch.cat(l1_parameters))
        # loss += l1

        loss.backward()
        optimizer.step()
        total_loss += loss

    model.eval()
    pred = model(test_wide_tensor.to(device))
    auc = roc_auc_score(test_tensor.cpu().detach().numpy(), pred.cpu().detach().numpy())
    auc_wide.append(auc)

    end = time.time()
    if epoch >= 0:
        print(f'EPOCH:{epoch + 1} === loss:{total_loss:.4f} === AUC:{auc:.4f} ===Elapsed:{(end - start):.4f}')
        torch.save(model.state_dict(),f'checkpoints/Wide_Model_Epoch_{epoch+1}_AUC_{auc:.4f}')
print(f'Wide_Model_Best_AUC:{sorted(auc_wide)[-1]:.4f}')

########################### " Deep Model " #################################
# For Deep
model_name = 'Deep'
train_dataset_deep = Deep_Dataset(X_deep_tensor=torch.LongTensor(X_train_deep), Y_tensor=torch.FloatTensor(Y_train))
train_loader_deep = DataLoader(dataset=train_dataset_deep, batch_size=config['batch_size'], shuffle=True)
model = Deep(df_deep).to(device)

optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0.05)
criterion = nn.BCELoss()
# criterion = nn.BCEWithLogitsLoss()
auc_deep = []
for epoch in range(config['epochs']):
    start = time.time()
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(train_loader_deep):
        X_deep, Y = batch[0], batch[1]
        X_deep, Y = X_deep.to(device), Y.to(device)
        optimizer.zero_grad()
        Y_pred = model(X_deep)
        loss = criterion(Y_pred.squeeze(), Y)

        # If you want to use L1 reg instead of l2 reg.
        # Set the adam's weight decay:0
        # Compute L1 loss
        # l1_weight =0.01
        # l1_parameters = []
        # for parameter in model.parameters():
        #     l1_parameters.append(parameter.reshape(-1))
        # l1 = l1_weight * model.compute_L1_loss(torch.cat(l1_parameters))
        # loss += l1

        loss.backward()
        optimizer.step()
        total_loss += loss

    model.eval()
    pred = model(test_deep_tensor.to(device))
    auc = roc_auc_score(test_tensor.cpu().detach().numpy(), pred.cpu().detach().numpy())
    auc_deep.append(auc)
    end = time.time()

    if epoch >= 0:
        print(f'EPOCH:{epoch + 1} === loss:{total_loss:.4f} === AUC:{auc:.4f} ===Elapsed:{(end - start):.4f}')
        torch.save(model.state_dict(),f'checkpoints/Deep_Model_Epoch_{epoch+1}_AUC_{auc:.4f}')
print(f'Deep_Model_Best_AUC:{sorted(auc_deep)[-1]:.4f}')