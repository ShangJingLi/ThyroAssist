import os
import subprocess
import signal
import sys
import cv2
import numpy as np
import gradio as gr
import mindspore
from mindspore import Tensor, context
from src.deep_learning.networks import UNet
from src.deep_learning.dataloader import download_and_unzip_unet_checkpoints

USE_ORANGE_PI = False
if os.name == 'nt':
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
else:
    try:
        if subprocess.run(['whoami'], capture_output=True, text=True, check=True).stdout.strip() == 'HwHiAiUser':
            context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', jit_config={"jit_level": "O2"})
            USE_ORANGE_PI = True
        else:
            context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
    except:
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False)

def on_terminate(signum, frame):
    if USE_ORANGE_PI:
        os.system('sudo npu-smi set -t pwm-duty-ratio -d 30')
        sys.exit(0)

signal.signal(signal.SIGINT, on_terminate)
signal.signal(signal.SIGTERM, on_terminate)

current_directory = os.getcwd()
target_directory = os.path.join(current_directory, 'unet_checkpoints')
if not os.path.exists(target_directory):
    download_and_unzip_unet_checkpoints()
else:
    pass
ckpt_file = os.path.join(target_directory, 'unet_checkpoints.ckpt')


net = UNet(n_channels=3, n_classes=2)
params = mindspore.load_checkpoint(ckpt_file)
mindspore.load_param_into_net(net, params)

if USE_ORANGE_PI:
    os.system('sudo npu-smi set -t pwm-duty-ratio -d 100')
    net(Tensor(np.zeros(shape=(1, 3, 572, 572)).astype(np.float32)))

input_data = gr.Image(label='请输入甲状腺超声图像')
output_data = gr.Image(label="结节位置如下图所示")

def infer_ultrasound_image(image):
    # 图像读取由gradio框架自动进行，为RGB格式
    image = cv2.resize(np.array(image), dsize=(572, 572))
    copied_image = np.copy(image)
    if len(image.shape) == 3:
        input_tensor = Tensor(np.expand_dims(image.transpose((2, 0, 1)), axis=0)).astype(mindspore.float32) / 127.5 - 1
    else:
        input_tensor = Tensor(np.expand_dims(np.tile(image, (3, 1, 1)), axis=0)).astype(mindspore.float32) / 127.5 - 1
    output_tensor = net(input_tensor)
    output_as_numpy = np.argmax(output_tensor.asnumpy(), axis=1).astype(np.uint8) * 255
    output_as_numpy = output_as_numpy.reshape(388, 388)

    kernel = np.ones((3, 3), np.uint8)
    opened_output = cv2.morphologyEx(output_as_numpy, cv2.MORPH_OPEN, kernel)
    processed_output = cv2.morphologyEx(opened_output, cv2.MORPH_CLOSE, kernel)

    resized_output = cv2.resize(processed_output, dsize=(572, 572))
    contours, hierarchy = cv2.findContours(resized_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    res = cv2.drawContours(copied_image, contours, -1, (100, 255, 0), 1)
    return res


iface = gr.Interface(fn=infer_ultrasound_image,
                     inputs=input_data,
                     outputs=output_data,
                     title = "甲状腺超声结节区域检测器",
                     description = "选择甲状腺超声图像，通过图像分割和轮廓检测确定结节区域。")
iface.launch()
