import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mindspore.dataset as ds

import acl
import acllite_utils as utils
from acllite_model import AclLiteModel
from acllite_resource import resource_list

import subprocess
import signal
import sys
import gradio as gr
import mindspore
from mindspore import Tensor, context
from src.deep_learning.networks import NestedUNet
from src.deep_learning.dataloader import download_and_unzip_nested_unet_checkpoints, download_ultrasound_images


class AclLiteResource:
    """
    AclLiteResource
    """
    def __init__(self, device_id=0):
        self.device_id = device_id
        self.context = None
        self.stream = None
        self.run_mode = None
        
    def init(self):
        """
        init resource
        """
        print("init resource stage:")
        ret = acl.init()

        ret = acl.rt.set_device(self.device_id)
        utils.check_ret("acl.rt.set_device", ret)

        self.context, ret = acl.rt.create_context(self.device_id)
        utils.check_ret("acl.rt.create_context", ret)

        self.stream, ret = acl.rt.create_stream()
        utils.check_ret("acl.rt.create_stream", ret)

        self.run_mode, ret = acl.rt.get_run_mode()
        utils.check_ret("acl.rt.get_run_mode", ret)

        print("Init resource success")

    def __del__(self):
        print("acl resource release all resource")
        resource_list.destroy()
        if self.stream:
            print("acl resource release stream")
            acl.rt.destroy_stream(self.stream)

        if self.context:
            print("acl resource release context")
            acl.rt.destroy_context(self.context)

        print("Reset acl device ", self.device_id)
        acl.rt.reset_device(self.device_id)
        print("Release acl resource success")

# 初始化资源
acl_resource = AclLiteResource()
acl_resource.init()

path = os.getcwd()
model_path = os.path.join(path, "nested_unet.om")
model = AclLiteModel(model_path)

def infer_ultrasound_image(image):
    # 图像读取由gradio框架自动进行，为RGB格式
    image = np.array(image).astype(np.uint8)
    copied_image = np.copy(image)
    image = cv2.resize(np.array(image), dsize=(256, 256)).astype(np.float32)
    if len(image.shape) == 3:
        input_image = np.expand_dims(image.transpose((2, 0, 1)), axis=0) / 127.5 - 1
    else:
        input_image = np.expand_dims(np.tile(image, reps=(3, 1, 1)), axis=0) / 127.5 - 1

    print("input_image:", input_image.shape, input_image.dtype)
    result = model.execute([input_image, ])  # 输入需要一个list类型的东西,里面需要一个numpy型的东西
    print(type(result))
    res = result[0].argmax(axis=1)  # 形状为(1, 256, 256)
    output_image = res[0].astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    opened_output = cv2.morphologyEx(output_image, cv2.MORPH_OPEN, kernel)
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