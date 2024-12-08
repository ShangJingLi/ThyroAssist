import os
import time
import zipfile
import openi
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


def download_images():
    openi.download_file(repo_id="enter/nodule_segmentation", file="image_to_show.zip", cluster="NPU",
                                save_path=".",
                                force=False)
    zip_file_path = 'image_to_show.zip'

    # 检查ZIP文件是否存在
    if os.path.exists(zip_file_path):
        # 使用with语句打开ZIP文件，确保文件正确关闭
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # 解压ZIP文件到当前目录
            zip_ref.extractall('.')

        # 解压完成后删除ZIP文件
        os.remove(zip_file_path)
        print(f'文件 {zip_file_path} 已解压并删除。')
    else:
        print(f'文件 {zip_file_path} 不存在。')

    print('病理图片数据下载成功！')


def power(image, power_value):
    copied_image = np.copy(image)

    normalized_image = cv2.normalize(copied_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    normalized_image = np.power(normalized_image, power_value)
    copied_image = np.uint8(normalized_image * 255)

    return copied_image


def watershed_algorthm(image):
    b, g, r = cv2.split(image)

    beta = 0.7
    image_to_process = cv2.addWeighted(b, beta, r, 1-beta, 0)
    image_gray = power(image_to_process, 0.5)

    # 高斯滤波
    for i in range(3):
        image_gray = cv2.GaussianBlur(image_gray, (5, 5), 0)

    # 基于直方图的二值化处理
    _, thresh = cv2.threshold(image_gray, 200, 255, cv2.THRESH_BINARY_INV)
    # thresh = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 401, 0)


    # 做开操作，是为了除去白噪声
    kernel = np.ones((3, 3), dtype=np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 做膨胀操作，是为了让前景漫延到背景，让确定的背景出现
    sure_bg = cv2.dilate(opening, kernel, iterations=2)

    # 为了求得确定的前景，也就是注水处使用距离的方法转化
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # 归一化所求的距离转换，转化范围是[0, 1]
    ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknow = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg, connectivity=8)
    markers = markers + 1
    markers[unknow == 255] = 0

    copy_image = np.copy(image)
    # 分水岭算法
    markers = cv2.watershed(copy_image, markers)

    # 分水岭算法得到的边界点的像素值为-1
    copy_image[markers == -1] = [0, 0, 255]

    return copy_image


__all__ = ['segmentation_by_threshold',
           'cells_segmentation',
           'std_cleaner',
           'draw_counters',
           'get_time',
           'rename_jpg_files',
           'download_images',
           'watershed_algorthm']
