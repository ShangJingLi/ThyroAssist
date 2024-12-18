import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mindspore.dataset as ds

import acl
import acllite_utils as utils
from acllite_model import AclLiteModel
from acllite_resource import resource_list
from src.deep_learning.ascend_utils import AclLiteResource

import subprocess
import signal
import sys
import gradio as gr
import mindspore
from mindspore import Tensor, context
from src.deep_learning.networks import NestedUNet
from src.deep_learning.dataloader import download_and_unzip_nested_unet_checkpoints, download_ultrasound_images

# 初始化资源
acl_resource = AclLiteResource()
acl_resource.init()


# 加载模型
path = os.getcwd()
model_path = os.path.join(path, "nested_unet.om")
model = AclLiteModel(model_path)


@utils.display_time
def pre_process(image):
    """
    image preprocess
    """
    copied_image = np.copy(image)
    image = cv2.resize(np.array(image), dsize=(256, 256))
    if len(image.shape) == 3:
        input_image = np.expand_dims(image.transpose((2, 0, 1)), axis=0) / 127.5 - 1
    else:
        input_image = np.expand_dims(np.tile(image, reps=(3, 1, 1)), axis=0) / 127.5 - 1
    return input_image, copied_image

# 图片推理
@utils.display_time
def inference(image):
    """
    model inference
    """
    input_image, copied_image = pre_process(image)
    # show_images = show_data["data"].asnumpy()
    # mask_images = show_data["label"].reshape([1, 512, 512])
    result = model.execute([input_image, ])  # 输入需要一个list类型的东西,里面需要一个numpy型的东西
    res = result[0].argmax(axis=1)  # result是一个列表，里面只有一个元素是一个numpy数组，即模型输出转numpy型的东西
    return res 

def infer_ultrasound_image(image):
    # 图像读取由gradio框架自动进行，为RGB格式
    input_image, copied_image = pre_process(image)
    result = model.execute([input_image, ])
    res = result[0].argmax(axis=1)  # 形状为(1, 256, 256)

    kernel = np.ones((5, 5), np.uint8)
    opened_output = cv2.morphologyEx(res[0], cv2.MORPH_OPEN, kernel)
    processed_output = cv2.morphologyEx(opened_output, cv2.MORPH_CLOSE, kernel)

    resized_output = cv2.resize(processed_output, dsize=(572, 572))
    contours, hierarchy = cv2.findContours(resized_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    image_with_contour = cv2.drawContours(cv2.resize(copied_image, dsize=(572, 572)), contours, -1, (100, 255, 0), 1)
    return image_with_contour

input_data = gr.Image(label='请输入甲状腺超声图像')
output_data = gr.Image(label="结节位置如下图所示")

iface = gr.Interface(fn=infer_ultrasound_image,
                     inputs=input_data,
                     outputs=output_data,
                     title = "基于UNet++的甲状腺超声结节区域检测器",
                     description = "选择甲状腺超声图像，通过图像分割和轮廓检测确定结节区域。")
iface.launch()