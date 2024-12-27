import random
import numpy as np
import os
import cv2
import pandas as pd




red_color = [128, 0, 0]  # 农田/绿化
green_color = [0, 128, 0] # 房屋
yellow_color = [128, 128, 0]  # 湖泊
blue_color = [0, 0, 128]  # 山




class_clolor = {
    0: red_color, # "农田/绿化"
    1: green_color, # "房屋"
    2: yellow_color, # "湖泊"
    3: blue_color, # 山
}


gray_img_folder = r'train_and_test\train\input_gray'
mask_path = r'train_and_test\train\label\label.png'

mask_img = cv2.imread(mask_path)
mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)

w, h, chanel = mask_img.shape


from skimage import io
all_gray_img = []
for img_name in os.listdir(gray_img_folder):
    img_path = os.path.join(gray_img_folder, img_name)
    if 'Ndvi' in img_name:
        img_gray = cv2.imread(img_path , cv2.IMREAD_UNCHANGED)
    else:
        img = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    all_gray_img.append(img_gray)

data = []
val_data = [] # 测试用
for class_name, color in class_clolor.items():

    coordinates = np.where((mask_img[:, :, 0] == color[0]) &  # Red 
    (mask_img[:, :, 1] == color[1]) &  # Green
    (mask_img[:, :, 2] == color[2]))   # Blue

    x, y = coordinates[0], coordinates[1]

    count =0 
    for x_, y_ in zip(x, y):

        count+=1

        input_x = {}
        for index_, img in enumerate(all_gray_img, start=1):
            input_x[index_] = img[x_, y_]
            if index_ >= 12:
                input_x[index_] = img[x_, y_]*255.0 #是否取绝对值
            else:
                input_x[index_] = img[x_, y_]
        input_x['y'] = class_name
        
        # 8：2随机划分训练和测试像素
        if random.randint(1, 10) <= 8:
            data.append(input_x)
        else:
            val_data.append(input_x)
        print(f"train count:{len(data)}\ttest count:{len(val_data)}")
        if count > 100000:
            break
    print(f"{color}: {count}")
#    将字典列表转换为DataFrame
df = pd.DataFrame(data)

# 将DataFrame导出到Excel文件
df.to_excel('output_ndvi_train.xlsx', index=False)


val_df = pd.DataFrame(val_data)

# 将DataFrame导出到Excel文件
val_df.to_excel('output_ndvi_test.xlsx', index=False)