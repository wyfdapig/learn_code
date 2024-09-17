import pandas as pd
import numpy as np

# XGBoost
import xgboost as xgb
class XGBoost:
    def __init__(self):
        self.model = xgb.XGBRegressor()
    
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    

# 随机森林
from sklearn.ensemble import RandomForestRegressor
class RandomForest:
    def __init__(self):
        self.model = RandomForestRegressor()
    
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)


# CNN
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32*8, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.unsqueeze(1) # 添加一个channel维度
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # 展平
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x