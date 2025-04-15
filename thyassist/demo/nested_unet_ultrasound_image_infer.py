"""基于UNet++和支持向量机的甲状腺超声影像分析"""
import os
import subprocess
import signal
import sys
import cv2
import numpy as np
import onnxruntime as ort
import gradio as gr
from thyassist.machine_learning.dataloader import (download_svm_model,
                                                   load_svm_model,
                                                   extract_features_from_image)
from thyassist.machine_learning.utils import is_ascend_available, is_gpu_available
from launcher import get_project_root


download_dir = get_project_root()
my_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="cyan",
    neutral_hue="gray",
    text_size="md",
    spacing_size="md",
    radius_size="md"
)


if not os.path.exists(os.path.join(download_dir, "svm_models")):
    download_svm_model()
else:
    pass

judge_echo_intensity_model , judge_echo_intensity_scaler = load_svm_model(os.path.join(download_dir, "svm_models",
                                                                                       "judge_echo_intensity_model.pkl"),
                                                                          os.path.join(download_dir, "svm_models",
                                                                                       "judge_echo_intensity_scaler.pkl"))
judge_microcalcification_model, judge_microcalcification_scaler = load_svm_model(os.path.join(download_dir, "svm_models",
                                                                                              "judge_microcalcification_model.pkl"),
                                                                          os.path.join(download_dir, "svm_models",
                                                                                       "judge_microcalcification_scaler.pkl"))
judge_solidity_model, judge_solidity_scaler = load_svm_model(os.path.join(download_dir, "svm_models",
                                                                          "judge_solidity_model.pkl"),
                                                                          os.path.join(download_dir,
                                                                                       "svm_models", "judge_solidity_scaler.pkl"))

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
        from thyassist.machine_learning.dataloader import download_nested_unet_om
        print("✅ 使用 Ascend 硬件进行模型推理")
        USE_ACL = True
    except FileNotFoundError:
        # pylint: disable=unused-import
        from thyassist.machine_learning.dataloader import download_resnet_onnx
        USE_ACL = False
        print("⚠️ NPU环境依赖加载异常，回退到CPU环境进行推理")

elif is_gpu_available():
    # 使用 NVIDIA GPU进行模型推理
    from thyassist.machine_learning.dataloader import download_nested_unet_onnx
    selected_provider = 'TensorrtExecutionProvider'
    print("✅ 使用 NVIDIA GPU 推理")

else:
    # 无硬件加速，使用CPU
    from thyassist.machine_learning.dataloader import download_nested_unet_onnx
    print("⚠️ 无可用 GPU/NPU，回退到 CPU 进行推理")


# 若进程终止，将风扇速度降低
def on_terminate(signum, frame):
    if USE_ORANGE_PI:
        os.system('sudo npu-smi set -t pwm-duty-ratio -d 30')
    sys.exit(0)

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
    model_path = os.path.join(download_dir, "nested_unet.om")
    if not os.path.exists(model_path):
        download_nested_unet_om()
    model = AclLiteModel(model_path)
else:
    # 非香橙派环境使用checkpoints进行推理
    model_path = os.path.join(download_dir, "nested_unet.onnx")
    if not os.path.exists(model_path):
        download_nested_unet_onnx()

    if is_gpu_available():
        cache_dir = os.path.join(download_dir, 'trt_cache', 'nested_unet')
        os.makedirs(cache_dir, exist_ok=True)

        os.environ['ORT_TENSORRT_ENGINE_CACHE_ENABLE'] = '1'
        os.environ['ORT_TENSORRT_CACHE_PATH'] = cache_dir

        has_cache = any(fname.endswith('.engine') for fname in os.listdir(cache_dir))

        if not has_cache:
            print(f"🛠️ 检测到首次使用模型 nested_unet.onnx，正在构建 TensorRT 引擎缓存...")
        else:
            print(f"✅ 已检测到模型 nested_unet.onnx 的 TensorRT 缓存，将直接加载。")

    session = ort.InferenceSession(model_path, providers=[selected_provider])

# 定义gradio的Interface类
def infer_ultrasound_image(image):
    if image.shape == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    copied_image = np.copy(cv2.resize(image, dsize=(572, 572)))
    image = cv2.resize(image, dsize=(256, 256))

    if USE_ACL:
        context, _ = acl.rt.get_context()
        if context != acl_resource.context:
            acl.rt.set_context(acl_resource.context)
        input_array = np.expand_dims(image.astype(np.float32).transpose((2, 0, 1)), axis=0) / 127.5 - 1
        result = model.execute([input_array])
        output_as_numpy = np.argmax(result[0], axis=1).astype(np.uint8) * 255
        output_as_numpy = output_as_numpy.reshape(256, 256)
    else:
        input_array = np.expand_dims(image.astype(np.float32).transpose((2, 0, 1)), axis=0) / 127.5 - 1
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        result = session.run([output_name], {input_name: input_array})
        output_as_numpy = np.argmax(result[0], axis=1).astype(np.uint8) * 255
        output_as_numpy = output_as_numpy.reshape(256, 256)

    kernel = np.ones((5, 5), np.uint8)
    opened_output = cv2.morphologyEx(output_as_numpy, cv2.MORPH_OPEN, kernel)
    processed_output = cv2.morphologyEx(opened_output, cv2.MORPH_CLOSE, kernel)
    resized_output = cv2.resize(processed_output, dsize=(572, 572))
    contours, _ = cv2.findContours(resized_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    result_image = cv2.drawContours(np.copy(copied_image), contours, -1, (100, 255, 0), 1)

    result = ""
    if len(contours[0]) == 0:
        result = "未发现结节"
        return result_image, result

    flag = 0
    # 计算结节区域水平和垂直的外界矩形
    x, y, w, h = cv2.boundingRect(contours[0])  # 获取水平垂直的外界矩形；x,y对应的点是image[y][x]
    roi = cv2.cvtColor(copied_image, cv2.COLOR_RGB2GRAY)[y:y + h, x:x + w]  # 边界矩形区域

    if h - w > 10:
        result += "纵横比大于1 "
        flag += 3
    else:
        result += "纵横比小于或近似等于1 "

    roi = cv2.resize(roi, dsize=(32, 32))
    roi_features = extract_features_from_image(roi)
    echo_intensity_features_scaled = judge_echo_intensity_scaler.transform([roi_features])
    microcalcification_features_scaled = judge_microcalcification_scaler.transform([roi_features])
    solidity_features_scaled = judge_solidity_scaler.transform([roi_features])

    echo_intensity_prediction = judge_echo_intensity_model.predict(echo_intensity_features_scaled)
    microcalcification_prediction = judge_microcalcification_model.predict(microcalcification_features_scaled)
    solidity_prediction = judge_solidity_model.predict(solidity_features_scaled)



    if solidity_prediction[0] == 0:
        result += "实性结节 "
        flag += 2
        if echo_intensity_prediction[0] == 0:
            flag += 2
            result += "低回声"
        else:
            flag += 1
    else:
        result += "囊性结节 "

    if microcalcification_prediction[0] == 0:
        result += "有微钙化现象 "
        flag += 1
    else:
        result += "无微钙化现象 "

    grade_list = ["TR1", "TR1", "TR2", "TR3", "TR4 A", "TR4 B", "TR4 C", "TR5"]
    advice_list = ["恶性可能≤2%，建议根据自身需求决定是否进一步检查",
              "恶性可能≤2%，建议根据自身需求决定是否进一步检查",
              "恶性可能≤2%，建议根据自身需求决定是否进一步检查",
              "恶性可能＜5%，建议根据结节大小进行随访或细针穿刺活检",
              "恶性风险2%-10%，建议根据结节大小进行随访或细针穿刺活检",
              "恶性风险10%-50%，建议进行随访或细针穿刺活检",
              "恶性可能约50%-90%，强烈建议进行随访或细针穿刺活检",
              "恶性可能＞90%，请立即进行细针穿刺活检"]

    grade1 = grade_list[min(flag, 7)]
    grade2 = grade_list[min(flag + 1, 7)]
    advice1 = advice_list[min(flag, 7)]
    advice2 = advice_list[min(flag + 1, 7)]


    if grade1 == grade2:
        result += f"初步定级为{grade1}, {advice1}"
    else:
        result += f"请观察结节外观，若边界清晰且规则，则初步定级为{grade1}，{advice1},若边界模糊且不规则，则初步定级为{grade2}，{advice2}"
    return result_image, result

with gr.Blocks(theme=my_theme) as demo:
    gr.Markdown("<h1 style='text-align: center;'>基于UNet++的甲状腺超声检测器</h1>")
    gr.Markdown("### 上传超声影像，获取结节区域信息和诊断建议。")
    gr.Markdown("---")
    with gr.Row():
        # 给图片组件设置自定义 CSS ID
        image_input = gr.Image(label="原始超声图片", elem_id="image_input_container")
        image_output = gr.Image(label="结节预测位置", elem_id="image_output_container")

    gr.Markdown("---")
    with gr.Row():
        text_output = gr.Textbox(label="svm预测报告", interactive=False, lines=2)

    gr.Markdown("---")

    with gr.Row():
        # 按钮触发处理逻辑
        button = gr.Button("Execute")
        button.click(infer_ultrasound_image,
                     inputs=[image_input],
                     outputs=[image_output, text_output])  # 输入原始超声图像，输出带轮廓超声图像及svm预测结果

        # 清除按钮
        clear_button = gr.ClearButton(
            components=[image_input, image_output, text_output],  # 清空错误信息
        )

demo.launch(inbrowser=True)
