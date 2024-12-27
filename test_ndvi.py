import numpy as np
import os
import cv2
import torch
import pandas as pd
from train_ndvi import Net, feature_number, out_prediction
from svm_train import svm_predict_single
from juceshu_train import decision_tree_predict_single


# 首先设定模型的保存路径
PATH = 'models/net_ndvi_nonabs.pth'


# 首先实例化模型的类对象
net = Net(n_feature=feature_number,
                    n_output=out_prediction,
                    n_layer=5,
                    n_neuron1=100,
                    n_neuron2=100) # 这里直接确定了隐藏层数目以及神经元数目，实际操作中需要遍历
# 加载训练阶段保存好的模型的状态字典
net.load_state_dict(torch.load(PATH))
net.eval()



red_color = [128, 0, 0]  # 农田/绿化
green_color = [0, 128, 0] # 房屋
yellow_color = [128, 128, 0]  #山
blue_color = [0, 0, 128]  # "湖泊" 

# 标注时的类别和颜色对应关系
# train_class_clolor = {
#     0: red_color, # "农田/绿化"
#     1: green_color, # "房屋"
#     2: yellow_color, # "湖泊"
#     3: blue_color, # 山
#     }


# 测试修改类别的颜色
class_clolor = {
    0: green_color, # 农田/绿化
    1: red_color, #  房屋
    2: yellow_color, #山
    3: blue_color, # 湖泊"
}


gray_img_folder = r'train_and_test\train\input_gray' # TODO 测试全图的话改成 input_gray\resized，测试分割后的train_and_test\test\input_gray
mask_path = r'ori\part.PNG' #  TODO  测试全图改成 ori\pan.png，测试分割后的train_and_test\test\label\test_img.png

mask_img = cv2.imread(mask_path)
mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)

w, h, chanel = mask_img.shape

# 读取14张输入图像放到all_gray_img
all_gray_img = []
for img_name in os.listdir(gray_img_folder):
    img_path = os.path.join(gray_img_folder, img_name)
    if 'Ndvi' in img_name:
        img_gray = cv2.imread(img_path , cv2.IMREAD_UNCHANGED)
    else:
        img = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    all_gray_img.append(img_gray)


# 初始化预测的mask
#predict_mask = np.zeros_like(mask_img)
predict_mask_svm = np.zeros_like(mask_img)
predict_mask_decision_tree = np.zeros_like(mask_img)
# ... 

h, w, c = img.shape 
batch_inputs_ori = np.zeros([1, 14])
batch_inputs = np.zeros([1, 14])

row_and_col = []
for row_ in range(h):
    for col_ in range(w):
        if list( mask_img[row_, col_, :] ) != [0, 0,0]:
        # if list( all_gray_img[0][row_, col_, :] ) != [0, 0,0]:
            input_x = []
            input_x_ori = []
            for index_, img in enumerate(all_gray_img, start=1):
                value = img[row_, col_]
                input_x_ori.append(value)


                if index_ >= 12: # ndvi读取的就是0~1
                    value = value  # 14列数据是否取绝对值
                else:
                    value = value/255.0
                input_x.append(value)

            inputs = np.asarray(input_x).reshape(1, -1)
            batch_inputs = np.concatenate((batch_inputs, inputs), axis=0)
            batch_inputs_ori = np.concatenate((batch_inputs_ori, np.asarray(input_x_ori).reshape(1, -1)), axis=0) # 没有除255的数据 因为svm训练时输入没有除以255
            
            row_and_col.append([row_, col_])
            if batch_inputs_ori.shape[0] == 100001 or (row_ == h-1 and col_ == w-1):
                print(f"{row_=}\t{col_=}")
                
                batch_inputs_ori = batch_inputs_ori[1:, :]
                batch_inputs = batch_inputs[1:, :]
                
                
                # 1 svm预测
                predicted_svm = svm_predict_single(batch_inputs_ori)
                for i in range(batch_inputs_ori.shape[0]):
                     row_, col_ = row_and_col[i]
                     predict_mask_svm[row_, col_, :] = class_clolor[predicted_svm[i]]
                #...

                # 2 ANN预测
                #X = torch.tensor(batch_inputs, dtype=torch.float32)  # 将数据集转换成torch能识别的格式
                #outputs = net(X)
                # 共有11个类别, 采用模型计算出的概率最大的作为预测的类别
                #_, predicted = torch.max(outputs, 1)
                # print(f"预测：{ predicted.numpy()[0]}")
                #predicted = predicted.numpy()
                #for i in range(batch_inputs.shape[0]):
                   # row_, col_ = row_and_col[i]
                   # predict_mask[row_, col_, :] = class_clolor[predicted[i]]
                #...

                # 3.决策树预测
                predicted_decision_tree = decision_tree_predict_single(batch_inputs_ori)
                for i in range(batch_inputs_ori.shape[0]):
                     row_, col_ = row_and_col[i]
                     predict_mask_decision_tree[row_, col_, :] = class_clolor[predicted_decision_tree[i]]
                #...
                batch_inputs_ori = np.zeros([1, 14])
                batch_inputs = np.zeros([1, 14])
                row_and_col = []

# 1.ANN预测mask保存       
#predict_mask = cv2.cvtColor(predict_mask, cv2.COLOR_BGR2RGB)
#cv2.imwrite(r'train_and_test\test\label\predict_mask_ann.png', predict_mask)

# # 2.SVM预测mask保存
predict_mask_svm= cv2.cvtColor(predict_mask_svm, cv2.COLOR_BGR2RGB)
cv2.imwrite(r'train_and_test\test\label\predict_mask_svm.png', predict_mask_svm)

# # 3.决策树预测mask保存
predict_mask_decision_tree= cv2.cvtColor(predict_mask_decision_tree, cv2.COLOR_BGR2RGB)
cv2.imwrite(r'train_and_test\test\label\predict_mask_decision_tree.png', predict_mask_decision_tree)



