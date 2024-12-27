import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from calculate_mask import predict_val


 
# X 是数据矩阵，每一行是一个样本，每一列是一个特征
# num_components 是要保留的主成分数量

# 加载数据 TODO
data = pandas.read_excel('output_ndvi_test.xlsx', header=None,index_col=None)

X = data.loc[:, 0:13]  # 将特征数据存储在x中，表格前14列为特征,
Y = data.loc[:, 14:14]  # 将标签数据存储在y中，表格最后一列为标签


X = X.iloc[1:].to_numpy()/255.0


Y = Y.iloc[1:].to_numpy().reshape(-1)


# 2. 标准化数据
# PCA对数据的尺度（scale）敏感，因此通常需要先标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 应用PCA
pca = PCA(n_components=2)  # 降到2维
principal_components = pca.fit_transform(X_scaled)
x, y  = principal_components[:,0], principal_components[:,1]





# 创建一个颜色字典，将每个类别映射到一个颜色
class_clolor = {
    0: "red", # "农田/绿化"
    1: "green", # "房屋"
    2: "yellow", # "湖泊"
    3: "blue", # 山
}
shape_dict = {
    0: "o", # "农田/绿化"
    1: "s", # "房屋"
    2: "*", # "湖泊"
    3: "^", # 山
}
 

# 创建一个包含2行2列子图的图形
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
# 使用plot函数画点
axs[0, 0].plot(x, y, 'o', c='black')
axs[0, 0].set_title("Original")


# 绘制深度学习点图
# 一、神经网络模型预测
y_test, y_pred = predict_val()
for label in list(shape_dict.keys()):
    color, shape = class_clolor[label], shape_dict[label]
    # 获取属于当前类别的数据点
    x_points = x[y_pred == float(label)]
    y_points = y[y_pred == float(label)]
    
    # 绘制数据点
    axs[0, 1].scatter(x_points, y_points, c=color, label=f'Class {label}', marker=shape)


axs[0, 1].set_title("Predicted, epochs=100")






# 绘制svm预测结果
from svm_train import svm_predict
y_pred, y_true = svm_predict()
for label in list(shape_dict.keys()):
    color, shape = class_clolor[label], shape_dict[label]
    # 获取属于当前类别的数据点
    x_points = x[y_pred == float(label)]
    y_points = y[y_pred == float(label)]
    
    # 绘制数据点
    axs[1, 0].scatter(x_points, y_points, c=color, label=f'Class {label}', marker=shape)


axs[1, 0].set_title("SVM")



# 绘制决策树预测结果
from juceshu_train import decision_tree_predict
y_pred, y_true = decision_tree_predict()
for label in list(shape_dict.keys()):
    color, shape = class_clolor[label], shape_dict[label]
    # 获取属于当前类别的数据点
    x_points = x[y_pred == float(label)]
    y_points = y[y_pred == float(label)]
    
    # 绘制数据点
    axs[1, 1].scatter(x_points, y_points, c=color, label=f'Class {label}', marker=shape)


axs[1, 1].set_title("Decision Tree")




fig.savefig("分类结果散点图.png")