import os
import torch
import math
import warnings
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from model import XGBoost
from model import RandomForest
from model import CNN
from data_loader import MyDataset
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae
from sklearn.utils import shuffle
from configuration import args
from func import nextBatch,drawPlot
import time

# 忽略警告
warnings.filterwarnings('ignore')

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def denormalize(x):    
    '''
    反归一化
    '''    
    # 如果输入是 list，则先将其转换为 PyTorch 的 Tensor
    # 检查输入的形状
    # print(f"Input shape before denormalize: {np.array(x).shape if isinstance(x, list) else x.shape}")
    if isinstance(x, list):
        x = torch.tensor(x)

    # 如果输入是 numpy 数组，则将其转换为 PyTorch 张量
    elif isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        
    max_val, min_val = torch.max(x), torch.min(x)
    return x * (max_val - min_val) + min_val

def net_train(model, trainX, trainY, loss_fn, optimizer):
    model.train()
    for X, Y in nextBatch(trainX, trainY, batch_size=args.batch_size):
        # 将 numpy 数组转换为 PyTorch 张量
        X, Y = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
        if args.iscuda:
            x, y_true = X.to(device), Y.to(device)
        
        x = x.unsqueeze(1) # 添加一个channel维度
            
        y_hat = model(x)
        l = loss_fn(y_hat, y_true)
        optimizer.zero_grad(set_to_none=True)
        l.backward()
        optimizer.step()
    # 每一个epoch返回一次loss
    _, train_rmse, train_mae, train_loss = net_test(model, train_X, train_Y, loss_fn)
    
    return (train_rmse,train_mae,train_loss)

def net_test(model, testX, testY, loss_fn):
    model.eval()
    y_hats = []
    y_trues = []
    test_l_sum, batch_count = 0, 0
    with torch.no_grad():
        for X, Y in nextBatch(testX, testY, batch_size=args.batch_size):
            # 将 numpy 数组转换为 PyTorch 张量
            X, Y = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
            if args.iscuda:
                x,y_true = X.to(device), Y.to(device)
                
            x = x.unsqueeze(1) # 添加一个channel维度
                            
            y_hat = model(x) # 一个batch的输出
            test_l_sum += loss_fn(y_hat, y_true).item()
            batch_count += 1
            y_hats.append(y_hat.detach().cpu().numpy())
            y_trues.append(y_true.detach().cpu().numpy())
            # print(len(y_hat),len(y_true))
        y_hats = np.concatenate(y_hats) # 沿着维度0，也就是batch_size的维度拼接
        y_trues = np.concatenate(y_trues)
    y_trues = y_trues.reshape(-1,1)
    # print(y_hats.shape,y_trues.shape)
    y_hats = denormalize(y_hats)
    y_trues = denormalize(y_trues)
    # print(y_hats.shape,y_trues.shape)
    rmse_score,mae_score = math.sqrt(mse(y_trues, y_hats)), mae(y_trues, y_hats)   
    return (y_hats, rmse_score, mae_score, test_l_sum / batch_count)

if __name__ == '__main__':
    train_X, train_Y, test_X, test_Y = MyDataset().loading('/data2/NieShiqin/learn_code/sell/concrete.csv')
    print("数据集加载完毕！")
    
    # 模型定义
    xgb_model = XGBoost()
    rf_model = RandomForest()
    cnn_model = CNN()
    optimizer =  optim.Adam(params=cnn_model.parameters(),lr=args.lr)
    # 使用MSE反向传播
    loss_fn = nn.MSELoss()

    if torch.cuda.is_available():
        args.iscuda = True
        cnn_model = cnn_model.to(device)

    train_loss_list, test_loss_list = [],[]
    train_rmse_list, train_mae_list = [],[]
    test_rmse_list, test_mae_list = [],[]
    train_times = 0.0

    # 模型训练
    xgb_model.fit(train_X, train_Y)
    rf_model.fit(train_X, train_Y)
    print("XGBoost和随机森林模型训练完毕！")
    
    # CNN 训练
    print("CNN开始训练！")
    time_start = time.time()
    for i in range(args.epochs):
        print("=========epoch {}=========".format(i + 1))
        train_rmse, train_mae, train_loss = net_train(cnn_model, train_X, train_Y, loss_fn, optimizer)
        train_loss_list.append(train_loss)
        train_rmse_list.append(train_rmse)
        train_mae_list.append(train_mae)
        print('Epoch: {}, RMSE: {:.4f}, MAE: {:.4f}, Train Loss: {:.8f}'.format(
            i + 1, train_rmse, train_mae, train_loss))
        
        # 评估训练结果
        net_y_hat, test_rmse, test_mae, test_loss = net_test(cnn_model, test_X, test_Y, loss_fn)
        test_loss_list.append(test_loss)
        test_rmse_list.append(test_rmse)
        test_mae_list.append(test_mae)
        print('Epoch: {}, RMSE: {:.4f}, MAE: {:.4f}, Test Loss: {:.8f}'.format(
            i + 1, test_rmse, test_mae, test_loss))

    print("CNN模型训练完毕！")
    time_end = time.time()
    print(f"CNN模型训练时间: {time_end - time_start}s")
    
    metrics = [train_loss_list,test_loss_list,train_rmse_list,test_rmse_list,
        train_mae_list,test_mae_list]
    # metrics curve
    fname = "{}_lr{}_b{}_h{}_d{}_metrics.png".format(args.model,args.lr,
        args.batch_size,args.hidden_size,args.drop_prob)
    drawPlot(metrics, fname, ["loss","rmse","mae"]) 

    # 对测试集上的结果进行线性组合
    print("开始对测试集上的结果进行线性组合！")
    # print(test_Y.shape)
    weights = [0.4, 0.35, 0.25]
    net_y_hat = np.array(net_y_hat).squeeze()
    xgb_test_predict = xgb_model.predict(test_X)
    rf_test_predict = rf_model.predict(test_X)
    # xgb_test_predict = xgb_test_predict.reshape(-1,1)
    # rf_test_predict = rf_test_predict.reshape(-1,1)
    # print(net_y_hat.shape, xgb_test_predict.shape, rf_test_predict.shape)
    final_predict = xgb_test_predict * weights[0] + rf_test_predict * weights[1] + net_y_hat * weights[2]

    # 计算测试集上的误差
    mse_test = mse(test_Y, final_predict)
    print(f"混合模型在最终测试集上的MSE: {mse_test}")