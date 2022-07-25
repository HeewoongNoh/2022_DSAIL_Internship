#Import Library
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score, roc_curve, auc
from model import Wide_Deep,Wide,Deep
from dataset import df_wide_pro,df_deep,df_wide

# Setting seed
torch.manual_seed(2023)

# Device
device = 'gpu' if torch.cuda.is_available() else 'cpu'

# Data set setting
Y = np.array(df_deep['label'])
df_wide_pro.drop(['label'],axis=1,inplace=True)
df_deep.drop(['label'],axis=1,inplace=True)

length = 46361 # len(train_df)
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


# Load wide, deep, wide & deep model has best AUC score from training.
wide_deep_model = Wide_Deep().to(device)
state_dict_wd = torch.load('checkpoints/Wide_Deep_Model_Epoch_3_AUC_0.6923')
wide_deep_model.load_state_dict(state_dict_wd)

wide_model = Wide().to(device)
state_dict_wide = torch.load('checkpoints/Wide_Model_Epoch_23_AUC_0.6780')
wide_model.load_state_dict(state_dict_wide)

deep_model = Deep().to(device)
state_dict_deep = torch.load('checkpoints/Deep_Model_Epoch_3_AUC_0.6529')
deep_model.load_state_dict(state_dict_deep)


#For Plotting ROC Curve for each model
plt.figure(figsize=(10,8))
fpr = dict()
tpr = dict()
roc_auc = dict()
lw=1
colors = ['red','blue','green']
for idx, model in enumerate([wide_deep_model,wide_model,deep_model]):
    model.eval()
    if idx == 0:
        pred = model(test_wide_tensor.to(device),test_deep_tensor.to(device))
        fpr[idx],tpr[idx], _ =roc_curve(test_tensor.cpu().detach().numpy(), pred.cpu().detach().numpy())
        roc_auc = auc(fpr[idx],tpr[idx])
        plt.plot(fpr[idx],tpr[idx],color=colors[idx],lw=lw, label = 'Wide & Deep_Model_ROC curve (area = ' + str(round(roc_auc, 4)) + ')')

    if idx == 1:
        pred = model(test_wide_tensor.to(device))
        fpr[idx], tpr[idx], _ = roc_curve(test_tensor.cpu().detach().numpy(), pred.cpu().detach().numpy())
        roc_auc = auc(fpr[idx], tpr[idx])
        plt.plot(fpr[idx], tpr[idx], color=colors[idx], lw=lw,
                 label='Wide_Model_ROC curve (area = ' + str(round(roc_auc, 4)) + ')')

    if idx == 2:
        pred = model(test_deep_tensor.to(device))
        fpr[idx], tpr[idx], _ = roc_curve(test_tensor.cpu().detach().numpy(), pred.cpu().detach().numpy())
        roc_auc = auc(fpr[idx], tpr[idx])
        plt.plot(fpr[idx], tpr[idx], color=colors[idx], lw=lw,
                 label='Deep_Model_ROC curve (area = ' + str(round(roc_auc, 4)) + ')')

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

plt.title('User-Voucher Redemption Dataset ROC')
plt.legend(loc="lower right")
plt.show()