

# 计算混淆矩阵
import torch
import pandas
import torch.optim as optim
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score  
from sklearn.model_selection import train_test_split  
from sklearn.datasets import load_iris  
from sklearn.svm import SVC  

import numpy as np
import os
import cv2
import torch
import pandas as pd
from train_ndvi import Net, feature_number, out_prediction


# 首先设定模型的保存路径
PATH = r'models\net_ndvi_nonabs.pth'

# 首先实例化模型的类对象
net = Net(n_feature=feature_number,
                    n_output=out_prediction,
                    n_layer=5,
                    n_neuron1=100,
                    n_neuron2=100) # 这里直接确定了隐藏层数目以及神经元数目，实际操作中需要遍历
# 加载训练阶段保存好的模型的状态字典
net.load_state_dict(torch.load(PATH))
net.eval()


class_clolor = {
    0: [0,255,0], # "农田/绿化"
    1: [255,0,0], # "房屋"
    2: [193,210,240], # "湖泊"
    3: [255,255,0], # 山
}


def get_val_data():
    """
    测试excel --> numpy格式的 X, Y
    
    """
    
    # 加载数据 
    data = pandas.read_excel('output_ndvi_test.xlsx', header=None,index_col=None)

    X = data.loc[:, 0:13]  # 将特征数据存储在x中，表格前14列为特征,
    Y = data.loc[:, 14:14]  # 将标签数据存储在y中，表格最后一列为标签


    X = X.iloc[1:].to_numpy()

    Y = Y.iloc[1:].to_numpy().reshape(-1)

    Y = Y.astype('int')
    return  X, Y


def get_val_dataloader():

    # 加载数据 TODO
    X,Y = get_val_data()


    X = torch.tensor(X, dtype=torch.float32)  # 将数据集转换成torch能识别的格式
    Y = torch.tensor(Y.astype(float), dtype=torch.long)
    torch_dataset = torch.utils.data.TensorDataset(X, Y)  # 组成torch专门的数据库
    batch_size = 256  # 设置批次大小


    # Dataloader
    val_dataloader = torch.utils.data.DataLoader(torch_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)
    return val_dataloader



def predict_val():
    """
    测试深度学习算法
    """
    val_dataloader = get_val_dataloader()


    y_test = np.zeros(1)
    y_pred = np.zeros(1)
    for batch_idx, (inputs, labels) in enumerate(val_dataloader):

        
        inputs = inputs/255.0
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.numpy()
        y_test = np.concatenate((y_test, labels), axis=0)
        y_pred = np.concatenate((y_pred, predicted), axis=0)

    y_test = y_test[1:]
    y_pred = y_pred[1:]
    # 计算准确率  
    accuracy = accuracy_score(y_test, y_pred)  
    print("深度学习算法 Accuracy:", accuracy)  
    return y_test, y_pred

def cal_predict_mask(y_test, y_pred, mask_save_name:str='mask.png'):
    """
    y_test: 真实标签
    y_pred： 预测的类别标签
    """
    # 计算混淆矩阵  
    cm = confusion_matrix(y_test, y_pred)  
    print("Confusion Matrix:")  
    print(cm)  
    

    # 计算准确率  
    accuracy = accuracy_score(y_test, y_pred)  
    print("Accuracy:", accuracy)  
    
    # 计算每个类别的精准率、召回率和F1分数  
    precision = precision_score(y_test, y_pred, average=None)  
    recall = recall_score(y_test, y_pred, average=None)  
    f1 = f1_score(y_test, y_pred, average=None)  

    # 打印每个类别的精准率、召回率和F1分数  
    print("Precision per class:", precision)  
    print("Recall per class:", recall)  
    print("F1 Score per class:", f1)  

    import seaborn as sns  
    import matplotlib.pyplot as plt  
    
    # 绘制混淆矩阵  
    plt.figure(figsize=(10, 7))  
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  
    plt.xlabel('Predicted')  
    plt.ylabel('Truth')  
    plt.savefig(mask_save_name)



def main():
    # 神经网络预测
    y_test, y_pred = predict_val()
    # 2.计算混淆矩阵
    cal_predict_mask(y_test, y_pred, mask_save_name="ANN_mask.png")




  
if __name__ == "__main__":
    main()
