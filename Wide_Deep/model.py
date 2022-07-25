import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import df_wide,df_wide_pro,df_deep

#Configuration about embedding, data shape
num_sessions = len(df_deep)
num_promotions = len(df_deep.promotion_id)
num_age = len(np.unique(df_deep.user_age_level)) + 10
num_gender = len(np.unique(df_deep.user_gender)) + 10
num_purchase = len(np.unique(df_deep.user_purchase_level))



# Wide & Deep model
class Wide_Deep(nn.Module):
    def __init__(self):
        super().__init__()

        # Deep Model
        # For embedding for embedding features of df_deep dataset
        # Embedding_cols = ['session_id','promotion_id','user_age_level','user_gender','user_purchase_level']
        # Continuous features = ['voucher_min_spend', 'voucher_discount','user_trd__orders_cnt_hist', 'user_trd__actual_gmv_usd_hist','user_trd__orders_cnt_platform_discount_hist', 'user_trd__max_gmv_usd_hist','user_trd__avg_gmv_usd_hist','user_trd__min_gmv_usd_hist']
        self.session_embed_layer = nn.Embedding(num_sessions, embedding_dim=16)
        self.promotion_embed_layer = nn.Embedding(num_promotions, embedding_dim=16)
        self.age_embed_layer = nn.Embedding(num_age, embedding_dim=16)
        self.gender_embed_layer = nn.Embedding(num_gender, embedding_dim=16)
        self.purchase_embed_layer = nn.Embedding(num_purchase, embedding_dim=16)
        self.len_cont = 8
        # 308 comes from continuous features dim + sum of embedding features embedding dimensions, 8+300
        self.linear_layer_1 = nn.Linear(in_features=80+self.len_cont, out_features=64)
        self.linear_layer_2 = nn.Linear(in_features=64, out_features=32)
        self.linear_layer_3 = nn.Linear(in_features=32, out_features=16)

        # Wide and Deep Model
        self.linear_wide = nn.Linear(in_features=df_wide_pro.shape[1] + 16, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def compute_L1_loss(self, w):
        return torch.abs(w).sum()

    def forward(self, X_wide, X_deep):
        # Deep model
        session_embedding = self.session_embed_layer(X_deep[:, 0])
        promotion_embedding = self.promotion_embed_layer(X_deep[:, 1])
        age_embedding = self.age_embed_layer(X_deep[:, 2])
        gender_embedding = self.gender_embed_layer(X_deep[:, 3])
        purchase_embedding = self.purchase_embed_layer(X_deep[:, 4])
        continuous_features = X_deep[:, 5:]

        concat_deep = torch.cat(
            [session_embedding, promotion_embedding, age_embedding, gender_embedding, purchase_embedding,
             continuous_features], dim=1)

        concat_deep = self.linear_layer_1(concat_deep)
        concat_deep = F.relu(concat_deep)
        concat_deep = self.linear_layer_2(concat_deep)
        concat_deep = F.relu(concat_deep)
        concat_deep = self.linear_layer_3(concat_deep)
        output_deep = F.relu(concat_deep)

        # Wide and Deep Model

        concat_wide_deep = torch.cat([X_wide, output_deep], dim=1)
        logit = self.linear_wide(concat_wide_deep)
        output = self.sigmoid(logit)

        return output


#Wide model
class Wide(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_wide = nn.Linear(in_features=df_wide_pro.shape[1], out_features=1)
        self.sigmoid = nn.Sigmoid()

    def compute_L1_loss(self, w):
        return torch.abs(w).sum()

    def forward(self, X_wide):
        # Wide Model
        logit = self.linear_wide(X_wide)
        output = self.sigmoid(logit)

        return output


#Deep model
class Deep(nn.Module):
    def __init__(self):
        super().__init__()

        # Deep Model
        # For embedding for embedding features of df_deep dataset
        # Embedding_cols = ['session_id','promotion_id','user_age_level','user_gender','user_purchase_level']
        # Continuous features = ['voucher_min_spend', 'voucher_discount','user_trd__orders_cnt_hist', 'user_trd__actual_gmv_usd_hist','user_trd__orders_cnt_platform_discount_hist', 'user_trd__max_gmv_usd_hist','user_trd__avg_gmv_usd_hist','user_trd__min_gmv_usd_hist']
        self.session_embed_layer = nn.Embedding(num_sessions, embedding_dim=16)
        self.promotion_embed_layer = nn.Embedding(num_promotions, embedding_dim=16)
        self.age_embed_layer = nn.Embedding(num_age, embedding_dim=16)
        self.gender_embed_layer = nn.Embedding(num_gender, embedding_dim=16)
        self.purchase_embed_layer = nn.Embedding(num_purchase, embedding_dim=16)
        self.len_cont = 8
        # 308 comes from continuous features dim + sum of embedding features embedding dimensions, 8+300
        self.linear_layer_1 = nn.Linear(in_features=80+self.len_cont, out_features=64)
        self.linear_layer_2 = nn.Linear(in_features=64, out_features=32)
        self.linear_layer_3 = nn.Linear(in_features=32, out_features=16)
        self.linear_final = nn.Linear(in_features=16, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def compute_L1_loss(self, w):
        return torch.abs(w).sum()

    def forward(self, X_deep):
        # Deep model
        session_embedding = self.session_embed_layer(X_deep[:, 0])
        promotion_embedding = self.promotion_embed_layer(X_deep[:, 1])
        age_embedding = self.age_embed_layer(X_deep[:, 2])
        gender_embedding = self.gender_embed_layer(X_deep[:, 3])
        purchase_embedding = self.purchase_embed_layer(X_deep[:, 4])
        continuous_features = X_deep[:, 5:]

        concat_deep = torch.cat(
            [session_embedding, promotion_embedding, age_embedding, gender_embedding, purchase_embedding,
             continuous_features], dim=1)

        concat_deep = self.linear_layer_1(concat_deep)
        concat_deep = F.relu(concat_deep)
        concat_deep = self.linear_layer_2(concat_deep)
        concat_deep = F.relu(concat_deep)
        concat_deep = self.linear_layer_3(concat_deep)
        output_deep = F.relu(concat_deep)
        logit = self.linear_final(output_deep)
        output = self.sigmoid(logit)

        return output
