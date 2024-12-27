'''训练部分'''
import torch
import pandas
import torch.optim as optim

feature_number = 14  # 设置特征数目
out_prediction = 4  # 设置输出数目 类别数
learning_rate = 0.001  # 设置学习率
epochs = 100  # 设置训练代数


 
 
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_output, n_neuron1, n_neuron2,n_layer):  # n_feature为特征数目，这个数字不能随便取,n_output为特征对应的输出数目，也不能随便取
        self.n_feature=n_feature
        self.n_output=n_output
        self.n_neuron1=n_neuron1
        self.n_neuron2=n_neuron2
        self.n_layer=n_layer
        super(Net, self).__init__()
        self.input_layer = torch.nn.Linear(self.n_feature, self.n_neuron1) # 输入层
        self.hidden1 = torch.nn.Linear(self.n_neuron1, self.n_neuron2) # 1类隐藏层    
        self.hidden2 = torch.nn.Linear(self.n_neuron2, self.n_neuron2) # 2类隐藏
        self.predict = torch.nn.Linear(self.n_neuron2, self.n_output) # 输出层
 
    def forward(self, x):
        '''定义前向传递过程'''
        out = self.input_layer(x)
        out = torch.relu(out) # 使用relu函数非线性激活
        out = self.hidden1(out)
        out = torch.relu(out)
        for i in range(self.n_layer):
            out = self.hidden2(out)
            out = torch.relu(out) 
        out = self.predict( # 回归问题最后一层不需要激活函数
            out
        )  # 除去feature_number与out_prediction不能随便取，隐藏层数与其他神经元数目均可以适当调整以得到最佳预测效果
        return out

def load_train_data():
    # 加载数据 TODO
    data = pandas.read_excel('output_ndvi_train.xlsx', header=None,index_col=None)
    # data = pandas.read_excel('output_ndvi_test.xlsx', header=None,index_col=None)

    X = data.loc[:, 0:13]  # 将特征数据存储在x中，表格前14列为特征,
    Y = data.loc[:, 14:14]  # 将标签数据存储在y中，表格最后一列为标签


    X = X.iloc[1:].to_numpy()

    Y = Y.iloc[1:].to_numpy().reshape(-1)
    return X, Y
    
    
def train_model():
    # 加载数据 TODO
    X, Y = load_train_data()

    
    X = torch.tensor(X, dtype=torch.float32)  # 将数据集转换成torch能识别的格式
    Y = torch.tensor(Y.astype(float), dtype=torch.long)
    torch_dataset = torch.utils.data.TensorDataset(X, Y)  # 组成torch专门的数据库
    batch_size = 256  # 设置批次大小
    
    # 划分训练集测试集与验证集
    torch.manual_seed(seed=2021) # 设置随机种子分关键，不然每次划分的数据集都不一样，不利于结果复现
    train_validaion, test = torch.utils.data.random_split(
        torch_dataset,
        [210000, X.size()[0]-210000 ],
    )  # 先将数据集拆分为训练集+验证集（共450组），测试集（56组） 222832 
    train, validation = torch.utils.data.random_split(
        train_validaion, [200000, 10000])  # 再将训练集+验证集拆分为训练集400，测试集50
    
    # 再将训练集划分批次，每batch_size个数据一批（测试集与验证集不划分批次）
    train_data = torch.utils.data.DataLoader(train,
                                            batch_size=batch_size,
                                            shuffle=True)



    net = Net(n_feature=feature_number,
                        n_output=out_prediction,
                        n_layer=5,
                        n_neuron1=100,
                        n_neuron2=100) # 这里直接确定了隐藏层数目以及神经元数目
    # optimizer = optim.Adam(net.parameters(), learning_rate)  # 使用Adam算法更新参数

    # net.cuda()
    import torch.nn as nn
    import torch.optim as optim
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(epochs):  # loop over the dataset multiple times
    
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_data):
            inputs = inputs/255.0
            # 首先将优化器梯度归零
            optimizer.zero_grad()
    
            # 输入图像张量进网络, 得到输出张量outputs
            outputs = net(inputs)
    
            # 利用网络的输出outputs和标签labels计算损失值
            loss = criterion(outputs, labels)
    
            # 反向传播+参数更新
            loss.backward()
            optimizer.step()
    
            # 打印轮次和损失值
            running_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, batch_idx + 1, running_loss / 2000))
                running_loss = 0.0
    # 首先设定模型的保存路径
    PATH = './net_ndvi_nonabs.pth'
    # 保存模型的状态字典
    torch.save(net.state_dict(), PATH)
    print('Finished Training')



if __name__ == "__main__":
    train_model()