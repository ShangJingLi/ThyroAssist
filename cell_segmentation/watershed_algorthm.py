import cv2
import numpy as np
from power import power


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