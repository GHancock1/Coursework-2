import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from torchvision.transforms import ToTensor
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Function to fetch dataset from github and remove empty columns (such as the errors of the stellar age)
def getData():
    df = pd.read_csv("https://raw.githubusercontent.com/GHancock1/Coursework-2/main/KESR.csv", sep=",", header=0,)
    df.dropna(axis = 1, inplace=True, how = "all")
    df.drop(list(df.filter(regex = 'err')), axis = 1, inplace = True)
    return df

# Normalization function, using z-score normalization
def normalize(df):
    scaled = df.copy()
    for column in scaled.select_dtypes(include=np.number).columns:
        scaled[column] = (scaled[column] - scaled[column].mean()) / scaled[column].std()
    return scaled



# Functions for neural network in Q3, copied from Q2. 

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)



def split(X,y, size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=1, stratify=y)
    return X_train, X_test, y_train, y_test
    
def tensors(X_train, X_test, y_train, y_test):
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

def load_data(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, device):
    train_df = TensorDataset(X_train_tensor, y_train_tensor)
    test_df = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_df, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_df, batch_size=256, shuffle=False)

    pos_weight = (y_train_tensor.numel() - y_train_tensor.sum()) / y_train_tensor.sum()
    pos_weight = pos_weight.to(device)

    return train_loader, test_loader, train_df, test_df, pos_weight

def training(epochs, train_loader, model,loss_func, optimizer, device):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device) # sends the tensors to the GPU if possible to speed up processing time
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_func(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * yb.size(0)

        model.eval()
    return model

def testing(model, X_test_tensor, y_test_tensor, device):
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_tensor.to(device))
        test_probs = torch.sigmoid(test_logits).cpu().numpy().ravel()
        test_preds = (test_probs >= 0.5).astype(int)

        test_y = y_test_tensor.cpu().numpy().ravel()
        acc = accuracy_score(test_y, test_preds)
    return acc
