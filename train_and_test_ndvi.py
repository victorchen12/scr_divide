import os
import cv2


# gray_folder = r'input_gray\ori'
# save_folder = r'input_gray\resized'
# for im_name in os.listdir(gray_folder):


#     im = cv2.imread(os.path.join(gray_folder, im_name), cv2.IMREAD_GRAYSCALE)
#     h,w = im.shape

#     if h>7800:
#         im2 = cv2.resize(im, [7731, 7611])
#         cv2.imwrite(os.path.join(save_folder, im_name), im2)
#     else:
#         cv2.imwrite(os.path.join(save_folder, im_name), im)

# gray_folder = r'input_gray\resized'
# save_train_folder = r'train_and_test\train\input_gray'
# save_test_folder = r'train_and_test\test\input_gray'
# for im_name in os.listdir(gray_folder):

#     im = cv2.imread(os.path.join(gray_folder, im_name), cv2.IMREAD_GRAYSCALE)
#     h,w = im.shape
#     train_im = im[:5000, :]
#     cv2.imwrite(os.path.join(save_train_folder, im_name), train_im)

#     test_im = im[5000:, :]
#     cv2.imwrite(os.path.join(save_test_folder, im_name), test_im)


# ndvi
import os
import cv2
import numpy as np
from torchvision import transforms






from PIL import Image
import numpy as np
from skimage import io

gray_folder = r'input_gray\ndvi'
save_train_folder = r'train_and_test\train\input_gray'
save_test_folder = r'train_and_test\test\input_gray'
for im_name in os.listdir(gray_folder):
    # img_path = os.path.join(gray_folder, im_name)

    # img = cv2.imread(img_path , cv2.IMREAD_UNCHANGED)#uint16
    # # img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    # # 显示图像
    # img[np.isnan(img)] = 0
 
    # # cv2.imshow('32-bit Image', img)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()
    # tensor_from_image= transforms.ToTensor()(img_RGB)

    # img_from_tensor = tensor_from_image.numpy().transpose((1, 2, 0))
    # img_from_tensor = img_from_tensor.astype(np.uint16)
    # img_from_tensor = cv2.cvtColor(img_from_tensor, cv2.COLOR_BGR2RGB)

    # cv2.imwrite('tensor_cv2.tif',img_from_tensor)

    # img = cv2.imread('tensor_cv2.tif')
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite(os.path.join(save_train_folder, im_name), img_gray[:5000, :])    

    # cv2.imwrite(os.path.join(save_test_folder, im_name), img_gray[5000:, :])    



    pil_image = Image.open(os.path.join(gray_folder, im_name))
    # im_name = im_name.replace('TIF', 'PNG')
    h,w  = pil_image.size

    
    # im = np.array(pil_image, dtype=np.float64)
    # h,w = im.shape
    train_im = pil_image.crop((0,0,7611, 5000))#im[:5000, :]
    train_im.save(os.path.join(save_train_folder, im_name))
    # cv2.imwrite(os.path.join(save_train_folder, im_name), train_im)

    test_im =  pil_image.crop((0,5000,7611, 7731))
    test_im.save(os.path.join(save_test_folder, im_name))
    # cv2.imwrite(os.path.join(save_test_folder, im_name), test_im)


# im = cv2.imread('ori\pan.png')
# h,w,c = im.shape
# train_im = im[:5000, :, :]
# cv2.imwrite(r"train_and_test\train_img.png", train_im)

# test_im = im[5000:, :, :]
# cv2.imwrite(r"train_and_test\test_img.png", test_im)


