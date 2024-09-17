import math
import numpy as np
import matplotlib.pyplot as plt

def drawPlot(heights, fname, ylabel):
    """
    功能：绘制训练集上的准确率和测试集上的loss和acc变化曲线
    heights: 纵轴值列表
    fname：保存的文件名
    """
    plt.figure(figsize=(9, 6))
    x = [i for i in range(1,len(heights[0]) + 1)]
    # 绘制训练集和测试集上的loss变化曲线子图
    plt.xlabel("epoch")
    # 设置横坐标的刻度间隔
    plt.xticks([i for i in range(0,len(heights[0]) + 1,5)])
    
    axe1 = plt.subplot(2,2,1)
    plt.ylabel(ylabel[0])
    axe1.set_title("train and test loss")
    axe1.plot(x,heights[0],label="train")
    axe1.plot(x,heights[1],label="test")
    plt.legend()

    axe2 = plt.subplot(2,2,2)
    plt.ylabel(ylabel[1])
    axe2.set_title("train and test RMSE")
    axe2.plot(x,heights[2],label="train")
    axe2.plot(x,heights[3],label="test")
    plt.legend()

    axe3 = plt.subplot(2,2,3)
    plt.ylabel(ylabel[2])
    axe3.set_title("train and test MAE")
    axe3.plot(x,heights[4],label="train")
    axe3.plot(x,heights[5],label="test")
    plt.legend()

    plt.savefig("images/{}".format(fname))
    plt.show()


def nextBatch(Xdata, Ydata, batch_size):
    """
    把数据分成batch_size大小的数据块
    """
    data_length = len(Ydata)
    num_batches = math.ceil(data_length / batch_size) # 向上取整
    for idx in range(num_batches): # idx从0开始，所以batch也是从0开始
        start_idx = batch_size * idx
        end_idx = min(start_idx + batch_size, data_length)
        yield Xdata[start_idx:end_idx], Ydata[start_idx:end_idx]

if __name__ == "__main__":
    pass