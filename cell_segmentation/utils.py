import os
import time
import cv2
import numpy as np


H_HIGH = 160
H_LOW = 120

S_HIGH = 190
S_LOW = 75

V_HIGH = 200
V_LOW = 65

B_HIGH = 210
B_LOW = 65

G_HIGH = 110
G_LOW = 50

R_HIGH = 210
R_LOW = 90


def segmentation_by_threshold(image:np.array, low:int, high:int):
    """根据设定的阈值对细胞进行双阈值分割"""
    if len(image.shape) != 2:
        raise TypeError("请输入灰度图像！")

    _, thresh1 = cv2.threshold(image, high, 255, cv2.THRESH_BINARY_INV)  # 低的值设为白，即设置上界
    _, thresh2 = cv2.threshold(image, low, 255, cv2.THRESH_BINARY)

    thresh = np.logical_and(thresh1 != 0, thresh2 != 0)

    return thresh


def cells_segmentation(image: np.array):
    """在多个通道对细胞执行双阈值分割并逐像素进行逻辑和运算"""
    if len(image.shape) != 3:
        raise TypeError("请输入彩色图像！")

    clone_image = std_cleaner(image)

    # 将bgr图像转hsv图像
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_clone_image = cv2.cvtColor(clone_image, cv2.COLOR_BGR2HSV)

    # 将图像分成6个通道
    b, g, r = cv2.split(image)
    _, s, v = cv2.split(hsv_image)

    ch, _, _ = cv2.split(hsv_clone_image)

    # 每个通道做一次阈值分割
    h_thresh = segmentation_by_threshold(ch, H_LOW, H_HIGH).astype(np.bool)
    s_thresh = segmentation_by_threshold(s, S_LOW, S_HIGH).astype(np.bool)
    v_thresh = segmentation_by_threshold(v, V_LOW, V_HIGH).astype(np.bool)

    b_thresh = segmentation_by_threshold(b, B_LOW, B_HIGH).astype(np.bool)
    g_thresh = segmentation_by_threshold(g, G_LOW, G_HIGH).astype(np.bool)
    r_thresh = segmentation_by_threshold(r, R_LOW, R_HIGH).astype(np.bool)

    thresh = h_thresh & s_thresh & v_thresh & b_thresh & g_thresh & r_thresh
    thresh = thresh.astype(np.uint8)
    ksize = (3, 3)
    # cv2.MORPH_CROSS 十字型	cv2.MORPH_RECT 矩形	   cv2.MORPH_ELLIPSE 椭圆形
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    thresh = cv2.erode(thresh, kernel, iterations=3)
    return thresh


def get_time(start: time.time, end:time.time):
    """传入两个time对象对代码段执行时间进行计时"""
    run_time = end - start
    if int(run_time // 3600) != 0:
        hours = f"{int(run_time // 3600)}小时"
    else:
        hours = ''
    if int(int(run_time) % 3600 // 60) != 0:
        minutes = f'{int(int(run_time) % 3600 // 60)}分钟'
    else:
        minutes = ''
    if int(run_time) % 60 != 0:
        seconds = f'{run_time % 60:.2f}秒'
    else:
        seconds = ''
    return hours + seconds + minutes


def rename_jpg_files(directory:str, start:int):
    """对目录下的.jpg文件进行批量改名"""
    # 确保提供的路径是一个目录
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a directory.")
        return
    # 获取目录下的所有文件和子目录列表
    files = os.listdir(directory)

    # 过滤出所有的.jpg文件
    jpg_files = [f for f in files if f.lower().endswith('.jpg')]

    # 对.jpg文件进行排序，以确保重命名时的顺序是正确的
    jpg_files.sort()

    # 对每个文件进行重命名
    for index, file_name in enumerate(jpg_files, start=start):
        # 构建完整的文件路径
        old_file_path = os.path.join(directory, file_name)
        # 构建新的文件路径
        new_file_path = os.path.join(directory, f"{index}.jpg")
        # 重命名文件
        os.rename(old_file_path, new_file_path)
        print(f"Renamed: {old_file_path} -> {new_file_path}")


def std_cleaner(image: np.array):
    """对RGB通道像素值相近的像素点进行去标准差操作，
       减少计算色调时的测量误差"""
    clone_image = np.copy(image)
    for i in range(clone_image.shape[0]):
        for j in range(clone_image.shape[1]):
            single_pixel = clone_image[i][j]
            if np.std(single_pixel) < 5:
                mean = np.mean(single_pixel)
                clone_image[i][j] = np.array([mean, mean, mean]).astype(np.uint8)
    return clone_image


def draw_counters(image:np.array, thresh:np.array):
    """对分割完成的二值图像进行轮廓绘制操作"""

    copied_image = np.copy(image)  # 复制原始图像作为轮廓背景
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)  # 设置逼近精度
        approx = cv2.approxPolyDP(contour, epsilon, True)  # 用多边形逼近轮廓，降低复杂度
        length = cv2.arcLength(approx, True)  # 计算轮廓长度

        # 筛选轮廓:根据周长平方和面积之比筛选轮廓，确保所绘制的轮廓均只包含单个细胞
        if 90 < length < 180 and cv2.arcLength(approx, True) ** 2 / (4 * np.pi * cv2.contourArea(approx)) < 1.6:
            if length > 130 and cv2.arcLength(approx, True) ** 2 / (4 * np.pi * cv2.contourArea(approx)) > 1.4:
                pass
            else:
                hull = cv2.convexHull(approx)  # 取凸包作为绘制轮廓
                cv2.drawContours(copied_image, [hull], -1, (0, 255, 0), 2)  # 使用绿色轮廓线



__all__ = ['segmentation_by_threshold',
           'cells_segmentation',
           'std_cleaner',
           'draw_counters',
           'get_time',
           'rename_jpg_files']
