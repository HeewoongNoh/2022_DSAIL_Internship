# From Data_Preprocessing.ipynb you can get three.pkl files
'''
df_wide_final.pkl that has labeled encoded promotion id, less sparse that df_wide_dncoding_promotionID.pkl
df_wide_encodingID.pkl encoding promotion id, more sparse.
You can choose one of above two wide data for wide part
df_deep_final.pkl for  deep model part
'''
#Import Library
from torch.utils.data import Dataset
import pickle
#Read wide data
# load
with open('./data_files/df_wide_final.pkl', 'rb') as f:
    df_wide = pickle.load(f)

#Read wide that has one-hot-encodided promotion id dataset more spare, more dimensions
with open('./data_files/df_wide_encoding_promotionID.pkl','rb') as f:
    df_wide_pro = pickle.load(f)

#Read deep data
with open('./data_files/df_deep_final.pkl','rb') as f:
    df_deep = pickle.load(f)


#For Wide & Deep model
class WD_Dataset(Dataset):
    def __init__(self, X_wide_tensor, X_deep_tensor, Y_tensor):
        self.X_wide_tensor = X_wide_tensor
        self.X_deep_tensor = X_deep_tensor
        self.Y_tensor = Y_tensor

    def __len__(self):
        return self.X_wide_tensor.size(0)

    def __getitem__(self, idx):
        return self.X_wide_tensor[idx], self.X_deep_tensor[idx], self.Y_tensor[idx]

#For Wide only model
class Wide_Dataset(Dataset):
    def __init__(self, X_wide_tensor, Y_tensor):
        self.X_wide_tensor = X_wide_tensor
        self.Y_tensor = Y_tensor

    def __len__(self):
        return self.X_wide_tensor.size(0)

    def __getitem__(self, idx):
        return self.X_wide_tensor[idx], self.Y_tensor[idx]


#For Deep only model
class Deep_Dataset(Dataset):
    def __init__(self, X_deep_tensor, Y_tensor):
        self.X_deep_tensor = X_deep_tensor
        self.Y_tensor = Y_tensor

    def __len__(self):
        return self.X_deep_tensor.size(0)

    def __getitem__(self, idx):
        return self.X_deep_tensor[idx], self.Y_tensor[idx]


