import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from train_ndvi import load_train_data
from calculate_mask import get_val_data, cal_predict_mask

def train_svm():
    # 加载数据集

    X, y = load_train_data()

    
    # 将数据集拆分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 特征缩放（标准化）
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 创建 SVM 分类器（这里使用线性核）
    svm_classifier = SVC(kernel='linear', random_state=42)
    
    # 训练分类器
    svm_classifier.fit(X_train, y_train.astype('int'))


    # 保存SVM分类器到文件
    joblib_file = "svm_classifier.pkl"
    joblib.dump(svm_classifier, joblib_file)
    print(f"SVM分类器已保存到 {joblib_file}")

    # 保存 StandardScaler 实例到文件
    scaler_file = "models/svm_classifier_scaler.pkl"
    joblib.dump(scaler, scaler_file)
    print(f"StandardScaler 已保存到 {scaler_file}")




joblib_file = "models/svm_classifier.pkl"
scaler_file = "models/svm_classifier_scaler.pkl"
# 从文件加载SVM分类器
loaded_svm_classifier = joblib.load(joblib_file)
print("SVM分类器已从文件加载")
# 从文件加载 StandardScaler 实例
loaded_scaler = joblib.load(scaler_file)
print("SVM StandardScaler 已从文件加载")


def svm_predict_single(X_test):
    """
    svm预测
    """
    X_test = loaded_scaler.transform(X_test)
    y_pred = loaded_svm_classifier.predict(X_test)
    return y_pred

def svm_predict():
    # 对测试集进行预测
    X_test,y_true = get_val_data()
    X_test = loaded_scaler.transform(X_test)
    y_pred = loaded_svm_classifier.predict(X_test)

    accuracy = accuracy_score(y_true, y_pred)
    print(f"SVM算法准确率: {accuracy:.2f}")
    return y_pred, y_true

if __name__ == '__main__':
    # train_svm()
    y_pred, y_true = svm_predict()
    # 计算混淆矩阵
    cal_predict_mask(y_true, y_pred, mask_save_name="results/svm_predict_mask.png")
    
    