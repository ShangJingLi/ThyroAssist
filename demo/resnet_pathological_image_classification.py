import os
import subprocess
import signal
import sys
import cv2
import numpy as np
import gradio as gr
from src.machine_learning.dataloader import download_pathological_images, boundary_padding
from src.machine_learning.utils import is_ascend_available, is_gpu_available


USE_ORANGE_PI = False

# 后端类型判断及context设置
if os.name == 'nt':  # Windows操作系统, 使用CPU
    import mindspore
    from mindspore import Tensor, context
    from src.machine_learning.networks import resnet152
    from src.machine_learning.dataloader import download_and_unzip_resnet_checkpoints
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    print("使用CPU环境启动模型推理")
else:
    try:
        current_user = subprocess.run(['whoami'], capture_output=True, text=True, check=True).stdout.strip()
        if current_user == 'HwHiAiUser':  # 如果当前用户是HwHiAiUser
            import acl
            import acllite_utils as utils
            from acllite_model import AclLiteModel
            from acllite_resource import resource_list
            from src.machine_learning.dataloader import download_resnet_om
            USE_ORANGE_PI = True
            print("使用香橙派启动模型推理，模型格式为.om")
        elif is_ascend_available():  # 检测Ascend环境是否可用
            import mindspore
            from mindspore import Tensor, context
            from src.machine_learning.networks import resnet152
            from src.machine_learning.dataloader import download_and_unzip_resnet_checkpoints
            context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False)
            print("使用Ascend环境启动模型推理")
        elif is_gpu_available():  # 检测GPU环境是否可用
            import mindspore
            from mindspore import Tensor, context
            from src.machine_learning.networks import resnet152
            from src.machine_learning.dataloader import download_and_unzip_resnet_checkpoints
            context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False)
            print("使用GPU环境启动模型推理")
        else:  # 如果Ascend和GPU都不可用，回退到CPU
            import mindspore
            from mindspore import Tensor, context
            from src.machine_learning.networks import resnet152
            from src.machine_learning.dataloader import download_and_unzip_resnet_checkpoints
            context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
            print("使用CPU环境启动模型推理")
    except Exception as e:  # 捕获所有意外错误并回退到CPU
        print(f"Error detected: {e}")
        import mindspore
        from mindspore import Tensor, context
        from src.machine_learning.networks import resnet152
        from src.machine_learning.dataloader import download_and_unzip_resnet_checkpoints
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
        print("出现意外错误！使用CPU环境启动模型推理")

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
    download_pathological_images()

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
    model_path = os.path.join(current_directory, "medical_resnet.om")
    if not os.path.exists(model_path):
        download_resnet_om()
    model = AclLiteModel(model_path)
else:
    # 非香橙派环境使用checkpoints进行推理
    ckpt_path = os.path.join(current_directory, 'medical_resnet_checkpoints')
    if not os.path.exists(ckpt_path):
        download_and_unzip_resnet_checkpoints()
    ckpt_file = os.path.join(ckpt_path, 'medical_resnet_checkpoints.ckpt')

    net = resnet152()
    params = mindspore.load_checkpoint(ckpt_file)
    mindspore.load_param_into_net(net, params)

# 定义gradio的Interface类
def infer_pathological_image(image):
    image = boundary_padding(image, aim_size=(572, 572), padding=255)

    if USE_ORANGE_PI:
        context, _ = acl.rt.get_context()
        if context != acl_resource.context:
            acl.rt.set_context(acl_resource.context)
        input_array = np.expand_dims(image.astype(np.float32).transpose((2, 0, 1)), axis=0) / 127.5 - 1
        result = model.execute([input_array])
        if result[0][0] > result[0][1]:
            text = "图像中的细胞为癌细胞"
        else:
            text = "图像中的细胞为正常细胞"
    else:
        input_tensor = Tensor(np.expand_dims(image.transpose((2, 0, 1)), axis=0).astype(np.float32)) / 127.5 - 1
        output_tensor = net(input_tensor)
        if output_tensor[0][0] > output_tensor[0][1]:
            text = "图像中的细胞为癌细胞"
        else:
            text = "图像中的细胞为正常细胞"


    return text

# 定义Interface对象的输入输出类型
input_data = gr.Image(label='请输入甲状腺超声图像')
output_data = gr.Textbox(label="模型预测结果", interactive=False, lines=1)

iface = gr.Interface(
    fn=infer_pathological_image,
    inputs=input_data,
    outputs=output_data,
    title="基于ResNet152的甲状腺上皮细胞分类器",
    description="选择显微镜下甲状腺上皮细胞图片，通过模型预测是否癌变。"
)

iface.launch()
