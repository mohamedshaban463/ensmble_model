import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import category_encoders as ce
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

train_orig = pd.read_csv('/content/train.csv')
test_orig = pd.read_csv('/content/test.csv')

y_labels = train_orig['TARGET'].values
test_ids = test_orig['SK_ID_CURR'] if 'SK_ID_CURR' in test_orig.columns else np.arange(len(test_orig))

X_df = train_orig.drop(columns=['TARGET', 'SK_ID_CURR'], errors='ignore')
X_test_df = test_orig.drop(columns=['SK_ID_CURR'], errors='ignore')
combined = pd.concat([X_df, X_test_df], axis=0).reset_index(drop=True)

num_cols_raw = combined.select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols_raw:
    combined[col].fillna(combined[col].median(), inplace=True)

combined['CREDIT_TO_ANNUITY_RATIO'] = combined['AMT_CREDIT'] / (combined['AMT_ANNUITY'] + 1e-5)
combined['CREDIT_TO_GOODS_RATIO'] = combined['AMT_CREDIT'] / (combined['AMT_GOODS_PRICE'] + 1e-5)
combined['INCOME_TO_ANNUITY_RATIO'] = combined['AMT_INCOME_TOTAL'] / (combined['AMT_ANNUITY'] + 1e-5)
combined['INCOME_TO_CREDIT_RATIO'] = combined['AMT_INCOME_TOTAL'] / (combined['AMT_CREDIT'] + 1e-5)
combined['INCOME_PER_PERSON'] = combined['AMT_INCOME_TOTAL'] / (combined['CNT_FAM_MEMBERS'] + 1e-5)
combined['DAYS_EMPLOYED_PERCENT'] = combined['DAYS_EMPLOYED'] / (combined['DAYS_BIRTH'] + 1e-5)
combined['CAR_TO_BIRTH_RATIO'] = combined['OWN_CAR_AGE'] / (combined['DAYS_BIRTH'] / -365.25 + 1e-5)
combined['CAR_TO_EMPLOYED_RATIO'] = combined['OWN_CAR_AGE'] / (combined['DAYS_EMPLOYED'] / -365.25 + 1e-5)

ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
combined['EXT_SOURCES_PROD'] = combined['EXT_SOURCE_1'] * combined['EXT_SOURCE_2'] * combined['EXT_SOURCE_3']
combined['EXT_SOURCES_SUM'] = combined[ext_cols].sum(axis=1)
combined['EXT_SOURCES_MEAN'] = combined[ext_cols].mean(axis=1)
combined['EXT_SOURCES_STD'] = combined[ext_cols].std(axis=1).fillna(0)

combined['GROUP_EXT_MEAN_BY_ORGANIZATION'] = combined.groupby('ORGANIZATION_TYPE')['EXT_SOURCE_2'].transform('mean')
combined['GROUP_CREDIT_MEAN_BY_EDUCATION'] = combined.groupby('NAME_EDUCATION_TYPE')['AMT_CREDIT'].transform('mean')

num_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols:
    combined[col].fillna(combined[col].median(), inplace=True)

n_train = len(train_orig)
X_train_fe = combined.iloc[:n_train].copy()
X_test_fe = combined.iloc[n_train:].copy()

cat_cols = combined.select_dtypes(exclude=[np.number]).columns.tolist()
target_enc = ce.TargetEncoder(cols=cat_cols, smoothing=15.0)

X_train_fe[cat_cols] = target_enc.fit_transform(X_train_fe[cat_cols], y_labels)
X_test_fe[cat_cols] = target_enc.transform(X_test_fe[cat_cols])

scaler = QuantileTransformer(output_distribution='normal', random_state=42)
all_cols = X_train_fe.columns.tolist()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_fe), columns=all_cols)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_fe), columns=all_cols)

print("Features scaled and target encoded successfully!")

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_lgb_opt = np.zeros(len(X_train_scaled))
oof_cat_opt = np.zeros(len(X_train_scaled))
oof_xgb_opt = np.zeros(len(X_train_scaled))

pred_lgb_opt = np.zeros(len(X_test_scaled))
pred_cat_opt = np.zeros(len(X_test_scaled))
pred_xgb_opt = np.zeros(len(X_test_scaled))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_labels)):
    print(f"Training GBDTs Fold {fold + 1} on GPU...")
    X_tr, y_tr = X_train_scaled.iloc[train_idx], y_labels[train_idx]
    X_va, y_va = X_train_scaled.iloc[val_idx], y_labels[val_idx]

    model_lgb = lgb.LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.015,
        num_leaves=127,
        max_depth=9,
        subsample=0.75,
        colsample_bytree=0.65,
        reg_alpha=12.0,
        reg_lambda=15.0,
        device='gpu',
        random_state=42 + fold,
        verbose=-1
    )
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)]
    )
    oof_lgb_opt[val_idx] = model_lgb.predict_proba(X_va)[:, 1]
    pred_lgb_opt += model_lgb.predict_proba(X_test_scaled)[:, 1] / n_splits

    model_cat = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.015,
        depth=7,
        l2_leaf_reg=15.0,
        task_type='GPU',
        random_state=42 + fold,
        verbose=0
    )
    model_cat.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], early_stopping_rounds=150)
    oof_cat_opt[val_idx] = model_cat.predict_proba(X_va)[:, 1]
    pred_cat_opt += model_cat.predict_proba(X_test_scaled)[:, 1] / n_splits

    model_xgb = xgb.XGBClassifier(
        n_estimators=1500,
        learning_rate=0.015,
        max_depth=8,
        subsample=0.75,
        colsample_bytree=0.65,
        reg_alpha=12.0,
        reg_lambda=15.0,
        tree_method='hist',
        device='cuda',
        random_state=42 + fold
    )
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb_opt[val_idx] = model_xgb.predict_proba(X_va)[:, 1]
    pred_xgb_opt += model_xgb.predict_proba(X_test_scaled)[:, 1] / n_splits

print(f"LightGBM CV ROC-AUC: {roc_auc_score(y_labels, oof_lgb_opt):.5f}")
print(f"CatBoost CV ROC-AUC: {roc_auc_score(y_labels, oof_cat_opt):.5f}")
print(f"XGBoost CV ROC-AUC:  {roc_auc_score(y_labels, oof_xgb_opt):.5f}")

class TabularTransformerL0MultiDim(nn.Module):
    def __init__(self, num_features, embed_dims, nhead=2, num_layers=2):
        super(TabularTransformerL0MultiDim, self).__init__()
        self.embed_dims = embed_dims
        self.projection = nn.ModuleList([nn.Linear(1, d) for d in embed_dims])
        self.attn_dim = 8
        self.align_projection = nn.ModuleList([nn.Linear(d, self.attn_dim) for d in embed_dims])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.attn_dim,
            nhead=nhead,
            dim_feedforward=32,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(num_features * self.attn_dim, 128),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        embedded = []
        for i, proj in enumerate(self.projection):
            feat_col = x[:, i].unsqueeze(-1)
            feat_emb = proj(feat_col)
            feat_aligned = self.align_projection[i](feat_emb)
            embedded.append(feat_aligned.unsqueeze(1))
        x = torch.cat(embedded, dim=1)
        x = self.transformer(x)
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.fc(x))

feature_names = X_train_scaled.columns.tolist()
embed_dims = [8 if 'EXT_SOURCE' in col or 'RATIO' in col or 'PERCENT' in col or col in cat_cols else 2 for col in feature_names]

X_train_tensor = torch.tensor(X_train_scaled.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_labels, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_scaled.values, dtype=torch.float32)

oof_trans_l0_multidim = np.zeros(len(X_train_scaled))
pred_trans_l0_multidim = np.zeros(len(X_test_scaled))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_labels)):
    print(f"Training Level-0 Variable-Dim Tabular Transformer - Fold {fold + 1}...")
    X_tr, y_tr = X_train_tensor[train_idx], y_train_tensor[train_idx]
    X_va, y_va = X_train_tensor[val_idx], y_train_tensor[val_idx]
    train_dataset = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

    model = TabularTransformerL0MultiDim(num_features=X_train_scaled.shape[1], embed_dims=embed_dims).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    best_auc = 0
    best_preds = None
    for epoch in range(10):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_preds = model(X_va.to(device)).cpu().numpy()
            val_auc = roc_auc_score(y_va.numpy(), val_preds)
            if val_auc > best_auc:
                best_auc = val_auc
                best_preds = val_preds.flatten()
                torch.save(model.state_dict(), f"best_l0_trans_multidim_fold_{fold}.pth")

    oof_trans_l0_multidim[val_idx] = best_preds
    best_model = TabularTransformerL0MultiDim(num_features=X_train_scaled.shape[1], embed_dims=embed_dims).to(device)
    best_model.load_state_dict(torch.load(f"best_l0_trans_multidim_fold_{fold}.pth"))
    best_model.eval()
    with torch.no_grad():
        pred_trans_l0_multidim += best_model(X_test_tensor.to(device)).cpu().numpy().flatten() / n_splits

print(f"Tabular Transformer CV ROC-AUC: {roc_auc_score(y_labels, oof_trans_l0_multidim):.5f}")

class TabularNN(nn.Module):
    def __init__(self, input_dim):
        super(TabularNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))

oof_nn = np.zeros(len(X_train_scaled))
pred_nn = np.zeros(len(X_test_scaled))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_labels)):
    print(f"Training Tabular NN - Fold {fold + 1}...")
    X_tr, y_tr = X_train_tensor[train_idx], y_train_tensor[train_idx]
    X_va, y_va = X_train_tensor[val_idx], y_train_tensor[val_idx]
    train_dataset = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

    model = TabularNN(X_train_scaled.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_auc = 0
    best_preds = None
    for epoch in range(15):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_preds = model(X_va.to(device)).cpu().numpy()
            val_auc = roc_auc_score(y_va.numpy(), val_preds)
            scheduler.step(val_auc)
            if val_auc > best_auc:
                best_auc = val_auc
                best_preds = val_preds.flatten()
                torch.save(model.state_dict(), f"best_nn_fold_{fold}.pth")

    oof_nn[val_idx] = best_preds
    best_model = TabularNN(X_train_scaled.shape[1]).to(device)
    best_model.load_state_dict(torch.load(f"best_nn_fold_{fold}.pth"))
    best_model.eval()
    with torch.no_grad():
        pred_nn += best_model(X_test_tensor.to(device)).cpu().numpy().flatten() / n_splits

print(f"Deep Tabular NN CV ROC-AUC: {roc_auc_score(y_labels, oof_nn):.5f}")

OOF_train_opt = np.column_stack([oof_lgb_opt, oof_cat_opt, oof_xgb_opt, oof_nn, oof_trans_l0_multidim])
OOF_test_opt = np.column_stack([pred_lgb_opt, pred_cat_opt, pred_xgb_opt, pred_nn, pred_trans_l0_multidim])

OOF_train_tensor_opt = torch.tensor(OOF_train_opt, dtype=torch.float32)
y_train_tensor = torch.tensor(y_labels, dtype=torch.float32).unsqueeze(1)
OOF_test_tensor_opt = torch.tensor(OOF_test_opt, dtype=torch.float32)

oof_meta_trans_opt = np.zeros(len(OOF_train_opt))
pred_meta_trans_opt = np.zeros(len(OOF_test_opt))

class SelfAttentionMetaTransformer(nn.Module):
    def __init__(self, num_models=5, d_model=64, nhead=8, num_layers=3):
        super(SelfAttentionMetaTransformer, self).__init__()
        self.projection = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.15,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model * num_models, 32),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.projection(x)
        x = self.transformer(x)
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.fc(x))

print("Training Level-1 Self-Attention Meta-Transformer Stacker...")
for fold, (train_idx, val_idx) in enumerate(skf.split(OOF_train_opt, y_labels)):
    X_tr, y_tr = OOF_train_tensor_opt[train_idx], y_train_tensor[train_idx]
    X_va, y_va = OOF_train_tensor_opt[val_idx], y_train_tensor[val_idx]
    train_dataset = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    model = SelfAttentionMetaTransformer(num_models=5).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    best_auc = 0
    best_preds = None
    for epoch in range(20):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_preds = model(X_va.to(device)).cpu().numpy()
            val_auc = roc_auc_score(y_va.numpy(), val_preds)
            if val_auc > best_auc:
                best_auc = val_auc
                best_preds = val_preds.flatten()
                torch.save(model.state_dict(), f"best_meta_attention_stack_fold_{fold}.pth")

    oof_meta_trans_opt[val_idx] = best_preds
    best_model = SelfAttentionMetaTransformer(num_models=5).to(device)
    best_model.load_state_dict(torch.load(f"best_meta_attention_stack_fold_{fold}.pth"))
    best_model.eval()
    with torch.no_grad():
        pred_meta_trans_opt += best_model(OOF_test_tensor_opt.to(device)).cpu().numpy().flatten() / n_splits

final_cv = roc_auc_score(y_labels, oof_meta_trans_opt)
print(f"Level-1 Self-Attention Stacker CV ROC-AUC: {final_cv:.5f}")
