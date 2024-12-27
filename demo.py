"""
彩色转灰度
"""
import os
import cv2
import numpy as np
from torchvision import transforms

folder= r'input'
out_folder = r'input_gray'
for img_name in os.listdir(folder):

    img_path = os.path.join(folder, img_name)

    img = cv2.imread(img_path ,-1)#uint16
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float16)

    tensor_from_image= transforms.ToTensor()(img_RGB)

    img_from_tensor = tensor_from_image.numpy().transpose((1, 2, 0))
    img_from_tensor = img_from_tensor.astype(np.uint16)
    img_from_tensor = cv2.cvtColor(img_from_tensor, cv2.COLOR_BGR2RGB)

    cv2.imwrite('tensor_cv2.tif',img_from_tensor)

    img = cv2.imread('tensor_cv2.tif')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(out_folder, img_name),img_gray)