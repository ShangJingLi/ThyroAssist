import os
import cv2
import numpy as np
import joblib
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from skimage import io, color, feature
from sklearn.preprocessing import StandardScaler


# 1. 图像预处理：将图像转换为特征
def extract_features_from_image(image_path):
    # 读取图片并转换为灰度图
    image = io.imread(image_path)
    if image.ndim == 2:
        image_gray = image
    else:
        image_gray = color.rgb2gray(image)

    if image_gray.shape != (32, 32):
        cv2.resize(image_gray, dsize=(32, 32))

    # 提取HOG特征 (Histogram of Oriented Gradients)
    features = feature.hog(image_gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    return features


# 2. 加载数据集并提取特征
def load_dataset(image_dir, label_map):
    images = []
    labels = []

    for label, folder_name in label_map.items():
        folder_path = os.path.join(image_dir, folder_name)
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(folder_path, filename)
                features = extract_features_from_image(image_path)
                images.append(features)
                labels.append(label)
                print(images, labels)

    return np.array(images), np.array(labels)


# 3. 训练SVM模型
def train_svm_model(X_train, y_train):
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 使用SVM训练模型
    model = svm.SVC(kernel='linear', random_state=42)
    model.fit(X_train_scaled, y_train)

    return model, scaler


# 4. 保存模型
def save_model(model, scaler, model_filename="judge_microcalcification_model.pkl", scaler_filename="judge_microcalcification_scaler.pkl"):
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)


# 5. 评估模型
def evaluate_model(model, scaler, X_test, y_test):
    # 对测试集进行标准化
    X_test_scaled = scaler.transform(X_test)

    # 预测
    y_pred = model.predict(X_test_scaled)

    # 打印评估报告
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return y_pred


# 6. 主程序
if __name__ == "__main__":
    image_dir = "judge_microcalcificatioin_datasets"  # 数据集目录路径
    label_map = {0: 'yes', 1: 'no'}  # 标签映射:0表示有微钙化，1表示无微钙化

    # 加载数据集并提取特征
    X, y = load_dataset(image_dir, label_map)

    # 将数据集拆分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # 训练模型
    model, scaler = train_svm_model(X_train, y_train)

    # 保存模型
    save_model(model, scaler)

    # 评估模型
    print("Evaluating the model...")
    evaluate_model(model, scaler, X_test, y_test)
