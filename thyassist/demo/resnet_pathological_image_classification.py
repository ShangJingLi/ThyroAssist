"""基于UNet++和支持向量机的甲状腺超声影像分析"""
import os
import subprocess
import zipfile
import tempfile
import signal
import sys
import cv2
import numpy as np
import onnxruntime as ort
import gradio as gr
from thyassist.machine_learning.dataloader import boundary_padding, center_crop
from thyassist.machine_learning.utils import is_ascend_available, is_gpu_available
from launcher import get_project_root


my_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="cyan",
    neutral_hue="gray",
    text_size="md",
    spacing_size="md",
    radius_size="md"
)

method='crop'
download_dir = get_project_root()
print(download_dir)
USE_ORANGE_PI = False
USE_ACL = False
selected_provider = 'CPUExecutionProvider'

if is_ascend_available():
    # 使用昇腾硬件进行模型推理
    current_user = subprocess.run(['whoami'], capture_output=True, text=True, check=True).stdout.strip()
    if current_user == 'HwHiAiUser':  # 如果当前用户是HwHiAiUser
        USE_ORANGE_PI = True
    try:
        import acl
        import acllite_utils as utils
        from acllite_model import AclLiteModel
        from acllite_resource import resource_list
        from thyassist.machine_learning.dataloader import download_resnet_om
        print("✅ 使用 Ascend 硬件进行模型推理")
        USE_ACL = True
    except FileNotFoundError:
        from thyassist.machine_learning.dataloader import download_resnet_onnx
        print("⚠️ NPU环境依赖加载异常，回退到CPU环境进行推理")

elif is_gpu_available():
    # 使用 NVIDIA GPU进行模型推理
    from thyassist.machine_learning.dataloader import download_resnet_onnx
    selected_provider = 'TensorrtExecutionProvider'
    
    print("✅ 使用 NVIDIA GPU 推理")

else:
    # 无硬件加速，使用CPU
    from thyassist.machine_learning.dataloader import download_resnet_onnx
    print("⚠️ 无可用 GPU/NPU，回退到 CPU 进行推理")


# 若进程终止，将风扇速度降低
def on_terminate(signum, frame):
    if USE_ORANGE_PI:
        os.system('sudo npu-smi set -t pwm-duty-ratio -d 30')
    sys.exit(0)


def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps, axis=0)

signal.signal(signal.SIGINT, on_terminate)
signal.signal(signal.SIGTERM, on_terminate)

if USE_ACL:
    # 使用香橙派，定义pyacl相关组件
    if USE_ORANGE_PI:
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
    model_path = os.path.join(download_dir, "medical_resnet.om")
    if not os.path.exists(model_path):
        download_resnet_om(method=method)
    model = AclLiteModel(model_path)
else:
    # 非香橙派环境使用checkpoints进行推理
    model_path = os.path.join(download_dir, 'medical_resnet.onnx')
    if not os.path.exists(model_path):
        download_resnet_onnx(method=method)
        
    if is_gpu_available():
        cache_dir = os.path.join(download_dir, 'trt_cache', 'medical_resnet')
        os.makedirs(cache_dir, exist_ok=True)

        os.environ['ORT_TENSORRT_ENGINE_CACHE_ENABLE'] = '1'
        os.environ['ORT_TENSORRT_CACHE_PATH'] = cache_dir

        has_cache = any(fname.endswith('.engine') for fname in os.listdir(cache_dir))

        if not has_cache:
            print(f"🛠️ 检测到首次使用模型 medical_resnet.onnx，正在构建 TensorRT 引擎缓存...")
        else:
            print(f"✅ 已检测到模型 medical_resnet.onnx 的 TensorRT 缓存，将直接加载。")

    session = ort.InferenceSession(model_path, providers=[selected_provider])


# 定义gradio的Interface类
def infer_pathological_image(image, zip_file):
    text = None
    file_result = None

    if image is not None:
        if method == "pad":
            image = boundary_padding(image, aim_size=(572, 572), padding=255)
        else:
            image = center_crop(image, aim_size=(572, 572))

        if USE_ACL:
            context, _ = acl.rt.get_context()
            if context != acl_resource.context:
                acl.rt.set_context(acl_resource.context)
            input_array = np.expand_dims(image.astype(np.float32).transpose((2, 0, 1)), axis=0) / 127.5 - 1
            result = model.execute([input_array])
            softmax_result = softmax(result[0][0])
            if softmax_result[0] > 0.99:
                text = "影像中细胞为癌细胞的概率大于99%"
            elif softmax_result[0] < 0.01:
                text = "影像中细胞为癌细胞的概率小于1%"
            else:
                text = f"影像中的细胞为癌细胞的概率为{round(softmax_result[0] * 100, 2)}%"
        else:
            input_array = np.expand_dims(image.astype(np.float32).transpose((2, 0, 1)), axis=0) / 127.5 - 1
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            result = session.run([output_name], {input_name: input_array})  # 运行推理
            softmax_result = softmax(result[0][0])
            if softmax_result[0] > 0.99:
                text = "影像中细胞为癌细胞的概率大于99%"
            elif softmax_result[0] < 0.01:
                text = "影像中细胞为癌细胞的概率小于1%"
            else:
                text = f"影像中的细胞为癌细胞的概率为{round(softmax_result[0] * 100, 2)}%"

    if zip_file is not None:
        zip_path = zip_file.name
        if not zip_path.endswith('.zip'):
            text = text + "(批处理操作仅允许上传.zip格式压缩的文件！)" if text is not None else "(批处理操作仅允许上传.zip格式压缩的文件！)"
            return text, file_result

        # 解析 ZIP
        input_arrays = []
        filenames = []
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:  # 直接通过路径打开 ZIP
            for filename in sorted(zip_ref.namelist()):
                if filename.lower().endswith(".jpg"):
                    filenames.append(filename)
                    with zip_ref.open(filename) as image_file:
                        image_data = image_file.read()
                        image_vector = np.frombuffer(image_data, dtype=np.uint8)
                        image = cv2.imdecode(image_vector, cv2.IMREAD_COLOR)
                        # 预处理
                        if method == "pad":
                            image = boundary_padding(image, aim_size=(572, 572), padding=255)
                        else:
                            image = center_crop(image, aim_size=(572, 572))

                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        input_array = np.expand_dims(image.astype(np.float32).transpose((2, 0, 1)), axis=0) / 127.5 - 1
                        input_arrays.append(input_array)

        if USE_ACL:
            context, _ = acl.rt.get_context()
            if context != acl_resource.context:
                acl.rt.set_context(acl_resource.context)
            results = model.execute(input_arrays)
            with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8", suffix=".txt") as temp_file:
                for i in range(len(results)):
                    softmax_result = softmax(results[i][0])

                    if softmax_result[0] > 0.99:
                        p = "大于99%"
                    elif softmax_result[0] < 0.01:
                        p = "小于1%"
                    else:
                        p = f"为{round(softmax_result[0] * 100, 2)}%"
                    line = f"{filenames[i]}     影像中细胞为癌细胞的概率{p}"
                    temp_file.write(line + "\n")

            file_result = temp_file.name

        else:
            with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8", suffix=".txt") as temp_file:
                # 写入内容
                for i in range(len(input_arrays)):
                    input_name = session.get_inputs()[0].name
                    output_name = session.get_outputs()[0].name
                    result = session.run([output_name], {input_name: input_arrays[i]})
                    softmax_result = softmax(result[0][0])

                    if softmax_result[0] > 0.99:
                        p = "大于99%"
                    elif softmax_result[0] < 0.01:
                        p = "小于1%"
                    else:
                        p = f"为{round(softmax_result[0] * 100, 2)}%"
                    line = f"{filenames[i]}     影像中细胞为癌细胞的概率{p}"
                    temp_file.write(line + "\n")

            file_result = temp_file.name

    return text, file_result

with gr.Blocks(theme=my_theme) as demo:
    gr.Markdown("<h1 style='text-align: center;'>基于ResNet152的甲状腺结节细针穿刺细胞影像分析</h1>")
    gr.Markdown("### 上传甲状腺结节细针穿刺细胞影像，获取影像中细胞为癌细胞的概率。")
    gr.Markdown("---")

    with gr.Row():
        image_input = gr.Image(label="上传图像", type="numpy")
        text_output = gr.Textbox(label="图像分析结果", interactive=False)

    gr.Markdown("### 上传甲状腺结节细针穿刺细胞影像压缩文件，获取所有影像中细胞为癌细胞的概率。")
    gr.Markdown("---")

    with gr.Row():
        file_input = gr.File(label="上传文件")
        file_output = gr.File(label="处理后的文件")

    gr.Markdown("---")

    with gr.Row():
        process_button = gr.Button("Execute")

        process_button.click(
            fn=infer_pathological_image,
            inputs=[image_input, file_input],
            outputs=[text_output, file_output]
        )

        # 清除按钮
        clear_button = gr.ClearButton(
            components=[image_input, file_input, text_output, file_output],  # 清空错误信息
        )

demo.launch(inbrowser=True)
