"""
决策树
"""

import joblib
# 导入必要的库
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from train_ndvi import load_train_data
from calculate_mask import get_val_data, cal_predict_mask

def train():
    # 准备数据
    X, y = load_train_data()
    y = y.astype('int')

    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 创建和训练模型
    clf = DecisionTreeClassifier(random_state=42)  # 可以添加其他参数来调整模型
    clf.fit(X_train, y_train)

    # 评估模型
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"准确率: {accuracy:.2f}")

    # 保存决策树分类器到文件
    joblib_file = "models/DecisionTree_classifier.pkl"
    joblib.dump(clf, joblib_file)
    print(f"决策树分类器已保存到 {joblib_file}")


joblib_file = "models/DecisionTree_classifier.pkl"
# 从文件加载分类器
clf = joblib.load(joblib_file)
print("决策树分类器已从文件加载")

def decision_tree_predict_single(X_test):
    """
    决策树 预测单行
    """
    y_pred = clf.predict(X_test)
    return y_pred

def decision_tree_predict():
    """
    决策树 预测
    """


    # 对测试集进行预测
    X_test,y_true = get_val_data()  # 这里是封装好的加载测试数据的函数，这个函数读取了之前8：2划分的测试数据excel，得到numpy格式的X和Y
    y_true = y_true.astype('int')


    # 评估模型
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"决策树算法准确率: {accuracy:.2f}")

    return y_pred, y_true

if __name__ == '__main__':
    # train() # 模型已经保存起来了，不需要再训练
    # 1.决策树预测
    y_pred, y_true = decision_tree_predict() 
    # 2.输入预测结果和真实结果  计算混淆矩阵
    cal_predict_mask(y_true, y_pred, mask_save_name="results/decision_tree_predict_mask.png")
    