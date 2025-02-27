import math
import cv2
import numpy as np
import pandas as pd
import gradio as gr
from scipy import stats
from src.image_processor.image_utils import cells_segmentation, draw_contours
from src.machine_learning.dataloader import TITLE


css = """
#image_input_container {
    max-width: 600px;
    margin: auto;
}
"""

def process_image(image:np.array):
    # 将输入图像转换为灰度图
    threshold = cells_segmentation(image)

    image_with_contours, contours = draw_contours(image, threshold)

    # 创建一个空的DataFrame来存储轮廓数据
    data = np.zeros(shape=(len(contours), 17)).astype(np.float32)

    # 比例转换
    pixel_to_um = 1 / 4  # 每4像素等于1微米
    flag = 0

    # 对每个轮廓计算相关指标并存储在DataFrame中
    for contour in contours:
        # 计算面积Area
        area = cv2.contourArea(contour) * pixel_to_um ** 2  # 将面积转换为微米²

        # 计算周长Perim.
        perimeter = cv2.arcLength(contour, True) * pixel_to_um  # 将周长转换为微米

        # 最大径长Feret
        rect = cv2.minAreaRect(contour)
        width = rect[1][0]  # 计算矩形的对角线长度
        height = rect[1][1]
        max_diameter = max(width, height)
        feret = max_diameter * pixel_to_um

        # 计算最小外接矩形 (Bounding Rectangle)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(h, w) / min(h, w)  # 长宽比

        # 计算圆度Circ.
        circularlity = (4 * math.pi * area) / (perimeter ** 2) if perimeter != 0 else 0

        min_feret = min(h, w) * pixel_to_um  # 转换为微米
        max_feret = max(h, w)* pixel_to_um  # 转换为微米

        # 计算FeretX, FeretY和FeretAngle
        feret_x = feret_y = feret_angle = None
        if contour.shape[0] > 1:
            for i in range(len(contour)):
                for j in range(i + 1, len(contour)):
                    dist = np.linalg.norm(contour[i] - contour[j])
                    if dist == max_feret:  # 最大直径的两个点
                        feret_x = abs(contour[i][0][0] - contour[j][0][0]) * pixel_to_um
                        feret_y = abs(contour[i][0][1] - contour[j][0][1]) * pixel_to_um
                        dx = contour[j][0][0] - contour[i][0][0]
                        dy = contour[j][0][1] - contour[i][0][1]
                        feret_angle = np.degrees(math.atan2(dy, dx))  # 计算最大直径的角度
                        break

        # 计算其他指标
        mean_intensity = np.mean(image[y:y + h, x:x + w])  # 均值强度
        min_intensity = np.min(image[y:y + h, x:x + w])  # 最小强度
        max_intensity = np.max(image[y:y + h, x:x + w])  # 最大强度
        std_dev_intensity = np.std(image[y:y + h, x:x + w])  # 强度标准差
        median_intensity = np.median(image[y:y + h, x:x + w])  # 强度中位数

        # 提取指定区域
        region = image[y:y + h, x:x + w]

        # 计算区域内的众数
        mode_val, _ = stats.mode(region, axis=None)

        # 计算中心质心
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        # 计算圆度
        roundness = 4 * area / (np.pi * (max_feret ** 2))

        # 计算Solidity（实心度）
        hull = cv2.convexHull(contour)
        solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) != 0 else 0

        data[flag] = np.array([area, mean_intensity, std_dev_intensity, mode_val, min_intensity,
                               max_intensity, perimeter, circularlity, feret, median_intensity,
                               feret_x, feret_y, feret_angle, min_feret, aspect_ratio, roundness,
                               solidity]).astype(np.float32)
        flag += 1

    # 将数据转换为 pandas DataFrame
    df = pd.DataFrame(data, columns=TITLE)

    # 导出为 CSV 文件
    csv_file = "contour_data.csv"
    df.to_csv(csv_file, index=False)

    return image_with_contours, csv_file

with gr.Blocks(css=css) as demo:
    gr.Markdown("<h1 style='text-align: center;'>图像分割与阈值处理工具</h1>")
    gr.Markdown("### 上传病理图片后，调整阈值以获得分割结果。")
    gr.Markdown("---")

    with gr.Row():
        # 给图片组件设置自定义 CSS ID
        image = gr.Image(label="原始病理图片", elem_id="image")

    with gr.Row():
        image_with_contours = gr.Image(label="自动阈值分割图片", elem_id="image_with_contours")

    with gr.Row():
        output_features = gr.File()

    with gr.Row():
        button = gr.Button("Execute")
        button.click(process_image, inputs=[image], outputs=[image_with_contours, output_features])

# 启动应用
demo.launch()
