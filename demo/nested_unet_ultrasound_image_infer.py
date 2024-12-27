import os
import subprocess
import signal
import sys
import cv2
import numpy as np
import gradio as gr
from src.deep_learning.dataloader import download_ultrasound_images


USE_ORANGE_PI = False

# 后端类型判断及context设置
if os.name == 'nt':  # Windows操作系统, 使用CPU
    import mindspore
    from mindspore import Tensor, context
    from src.deep_learning.networks import NestedUNet
    from src.deep_learning.dataloader import download_and_unzip_nested_unet_checkpoints
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
else:  # 非Windows操作系统，优先使用Ascend，若无效则使用GPU
    try:
        current_user = subprocess.run(['whoami'], capture_output=True, text=True, check=True).stdout.strip()
        if current_user == 'HwHiAiUser':  # 如果当前用户是HwHiAiUser
            import acl
            import acllite_utils as utils
            from acllite_model import AclLiteModel
            from acllite_resource import resource_list
            from src.deep_learning.dataloader import download_nested_unet_om
            USE_ORANGE_PI = True
        else:  # 当前用户不是HwHiAiUser，优先使用Ascend环境
            import mindspore
            from mindspore import Tensor, context
            from src.deep_learning.networks import NestedUNet
            from src.deep_learning.dataloader import download_and_unzip_nested_unet_checkpoints
            context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False)
    except ImportError:  # 如果Ascend不可用或发生ImportError，则使用GPU
        try:
            import mindspore
            from mindspore import Tensor, context
            from src.deep_learning.networks import NestedUNet
            from src.deep_learning.dataloader import download_and_unzip_nested_unet_checkpoints

            context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False)
        except ImportError:  # 如果GPU也不可用，则回退到CPU
            import mindspore
            from mindspore import Tensor, context
            from src.deep_learning.networks import NestedUNet
            from src.deep_learning.dataloader import download_and_unzip_nested_unet_checkpoints
            context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# 若进程终止，将风扇速度降低
def on_terminate(signum, frame):
    if USE_ORANGE_PI:
        os.system('sudo npu-smi set -t pwm-duty-ratio -d 30')
    sys.exit(0)

signal.signal(signal.SIGINT, on_terminate)
signal.signal(signal.SIGTERM, on_terminate)

# 判断是否下载推理用图片
current_directory = os.getcwd()
image_path = os.path.join(current_directory, 'ultrasound_images_to_show')
if not os.path.exists(image_path):
    download_ultrasound_images()

if USE_ORANGE_PI:
    # 使用香橙派，定义pyacl相关组件
    os.system('sudo npu-smi set -t pwm-duty-ratio -d 100')

    class AclLiteResource:
        """ACL Lite Resource management class"""

        def __init__(self, device_id=0):
            self.device_id = device_id
            self.context = None
            self.stream = None
            self.run_mode = None

        def init(self):
            """Initialize resources"""
            print("Initializing resources...")
            ret = acl.init()
            utils.check_ret("acl.init", ret)

            ret = acl.rt.set_device(self.device_id)
            utils.check_ret("acl.rt.set_device", ret)

            self.context, ret = acl.rt.create_context(self.device_id)
            utils.check_ret("acl.rt.create_context", ret)

            self.stream, ret = acl.rt.create_stream()
            utils.check_ret("acl.rt.create_stream", ret)

            self.run_mode, ret = acl.rt.get_run_mode()
            utils.check_ret("acl.rt.get_run_mode", ret)
            print("Resources initialized successfully")

        def __del__(self):
            print("Releasing ACL resources...")
            resource_list.destroy()
            if self.stream:
                acl.rt.destroy_stream(self.stream)
            if self.context:
                acl.rt.destroy_context(self.context)
            acl.rt.reset_device(self.device_id)
            print("Resources released successfully")

    acl_resource = AclLiteResource()
    acl_resource.init()
    model_path = os.path.join(current_directory, "nested_unet.om")
    if not os.path.exists(model_path):
        download_nested_unet_om()
    model = AclLiteModel(model_path)
else:
    # 非香橙派环境使用checkpoints进行推理
    ckpt_path = os.path.join(current_directory, 'nested_unet_checkpoints')
    if not os.path.exists(ckpt_path):
        download_and_unzip_nested_unet_checkpoints()
    ckpt_file = os.path.join(ckpt_path, 'nested_unet_checkpoints.ckpt')

    net = NestedUNet(n_channels=3, n_classes=2, is_train=False)
    params = mindspore.load_checkpoint(ckpt_file)
    mindspore.load_param_into_net(net, params)

# 定义gradio的Interface类
def infer_ultrasound_image(image):
    copied_image = np.copy(image)
    image = cv2.resize(image, dsize=(256, 256))

    if USE_ORANGE_PI:
        context, _ = acl.rt.get_context()
        if context != acl_resource.context:
            acl.rt.set_context(acl_resource.context)
        input_array = np.expand_dims(image.astype(np.float32).transpose((2, 0, 1)), axis=0) / 127.5 - 1
        result = model.execute([input_array])
        output_as_numpy = np.argmax(result[0], axis=1).astype(np.uint8) * 255
        output_as_numpy = output_as_numpy.reshape(256, 256)
    else:
        input_tensor = Tensor(np.expand_dims(image.transpose((2, 0, 1)), axis=0).astype(np.float32)) / 127.5 - 1
        output_tensor = net(input_tensor)
        output_as_numpy = np.argmax(output_tensor.asnumpy(), axis=1).astype(np.uint8) * 255
        output_as_numpy = output_as_numpy.reshape(256, 256)

    kernel = np.ones((5, 5), np.uint8)
    opened_output = cv2.morphologyEx(output_as_numpy, cv2.MORPH_OPEN, kernel)
    processed_output = cv2.morphologyEx(opened_output, cv2.MORPH_CLOSE, kernel)
    resized_output = cv2.resize(processed_output, dsize=(572, 572))
    contours, _ = cv2.findContours(resized_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    result_image = cv2.drawContours(cv2.resize(copied_image, dsize=(572, 572)), contours, -1, (100, 255, 0), 1)
    return result_image

# 定义Interface对象的输入输出类型
input_data = gr.Image(label='请输入甲状腺超声图像')
output_data = gr.Image(label="结节位置如下图所示")

iface = gr.Interface(
    fn=infer_ultrasound_image,
    inputs=input_data,
    outputs=output_data,
    title="基于UNet++的甲状腺超声结节区域检测器",
    description="选择甲状腺超声图像，通过图像分割和轮廓检测确定结节区域。"
)

iface.launch()
