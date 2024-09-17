import torch
import numpy as np
import pandas as pd

class MyDataset():
    def __init__(self):
        super(MyDataset, self).__init__()
        
    def loading(self,data_path):
        '''
        预处理
        '''
        # 数据导入
        data = pd.read_csv(data_path)
        
        # 数据预处理
        data = data.dropna()
        data = data.drop_duplicates()
        
        # 归一化
        # 获取最大最小值
        self.max_val, self.min_val = np.max(data), np.min(data)
        data = (data - self.min_val) / (self.max_val - self.min_val)
        
        # 划分特征和标签
        data_X = data.iloc[:, :-1]
        data_Y = data.iloc[:, -1]
        
        # 分成训练集和测试集，训练集占80%，测试集占20%
        data_X = np.array(data_X)
        data_Y = np.array(data_Y)
        train_size = int(len(data_X) * 0.8)
        train_X = data_X[:train_size]
        train_Y = data_Y[:train_size]
        test_X = data_X[train_size:]
        test_Y = data_Y[train_size:]
        
        return train_X, train_Y, test_X, test_Y

if __name__ == "__main__":
    data_path = 'concrete.csv'
    dataset = MyDataset()
    trainX, trainY, testX, testY = dataset.loading(data_path)
    print("trainX:", trainX.shape)
    print("testX:", testX.shape)
    print("trainY:", trainY.shape)
    print("testY:", testY.shape)