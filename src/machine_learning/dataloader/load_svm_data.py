import os
import joblib
import cv2
import openi
from skimage import color, feature

def download_svm_model():
    openi.download_model(repo_id="enter/nodule_segmentation", model_name="svm_models", save_path=".")

# 1. 加载训练好的模型和标准化器
def load_svm_model(model_filename:str, scaler_filename:str):
    # 加载保存的SVM模型和标准化器
    model = joblib.load(model_filename)
    scaler = joblib.load(scaler_filename)
    return model, scaler


# 2. 提取图像特征（与训练时相同）
def extract_features_from_image(roi):
    # 读取图片并转换为灰度图
    # 确保图像大小为 32x32
    if type(roi) == str:
        roi = cv2.imread(roi, 0)
    if roi.ndim != 2:
        raise ValueError("Only allow input of gray image")
    if roi.shape != (32, 32):
        roi = cv2.resize(roi, (32, 32))

    # 提取HOG特征 (Histogram of Oriented Gradients)
    features = feature.hog(roi, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    return features

__all__ = ["download_svm_model",
           "load_svm_model",
           "extract_features_from_image"]
