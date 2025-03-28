import time
import subprocess
import mindspore
import numpy as np
from mindspore import nn
import onnxruntime as ort


class StopTimeMonitor(mindspore.Callback):
    def __init__(self, run_time):
        super(StopTimeMonitor, self).__init__()
        self.run_time = run_time

    def on_train_begin(self, run_context):
        cb_params = run_context.original_args()
        cb_params.init_time = time.time()  # 获取开始训练的时间

    def on_train_step_end(self, run_context):
        """每个step结束后执行的操作"""
        cb_params = run_context.original_args()
        epoch_num = cb_params.cur_epoch_num  # 获取epoch值
        step_num = cb_params.cur_step_num  # 获取step值
        loss = cb_params.net_outputs  # 获取损失值loss
        cur_time = time.time()  # 获取当前时间戳

        if (cur_time - cb_params.init_time) > self.run_time:
            # 当训练时间达到规定的停止时间时，停止训练
            train_time = get_time(cb_params.init_time, cur_time)
            print(f"训练中止！ 用时: {train_time}, 当前epoch: {epoch_num}, 当前: {step_num}, loss:{loss}")
            run_context.request_stop()


def get_time(start: time.time, end:time.time):
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
    return hours + minutes + seconds


def split_ckpt(file_path, chunk_size=80 * 1024 * 1024):
    """将文件分割为多个小文件，每个文件大小为chunk_size（默认为80MB）。"""
    with open(file_path, 'rb') as file:
        chunk_index = 0
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            chunk_filename = f"{file_path}.part{chunk_index:03d}"
            with open(chunk_filename, 'wb') as chunk_file:
                chunk_file.write(chunk)
            chunk_index += 1

    print(f"文件分割完成，生成了{chunk_index}个部分。")


def restore_ckpt(part_prefix, output_file):
    """将分割的小文件恢复为原始文件。"""
    chunk_index = 0
    with open(output_file, 'wb') as outfile:
        while True:
            chunk_filename = f"{part_prefix}.part{chunk_index:03d}"
            try:
                with open(chunk_filename, 'rb') as infile:
                    outfile.write(infile.read())
                chunk_index += 1
            except FileNotFoundError:
                break  # 没有更多的部分文件，退出循环

    print("文件恢复完成！")


def is_ascend_available():
    try:
        # 尝试执行 set_env.sh 脚本以配置 Ascend 环境
        result = subprocess.run(
            ['/bin/bash', '-c', 'source /usr/local/Ascend/ascend-toolkit/set_env.sh && env'],  # 使用 `env` 检查环境变量
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        # 检查命令的返回码
        if result.returncode == 0:
            # 进一步检查是否设置了 Ascend 相关的环境变量
            if "ASCEND_HOME" in result.stdout or "ASCEND_OPP_PATH" in result.stdout:
                return True
    except (FileNotFoundError, Exception):
        # 捕获所有异常但不输出任何信息
        pass

    return False


def is_gpu_available():
    available_providers = ort.get_available_providers()
    return "TensorrtExecutionProvider" in available_providers


def export_ms_model(net:nn.Cell, model_name:str, input_shape:tuple,checkpoint_file_path:str, file_format:str):
    params = mindspore.load_checkpoint(checkpoint_file_path)
    mindspore.load_param_into_net(net, params)
    input_tensor = mindspore.Tensor(np.ones(shape=input_shape, dtype=np.float32))
    mindspore.export(net, input_tensor, file_name=model_name, file_format=file_format)


__all__ = ['get_time',
           'split_ckpt',
           'restore_ckpt',
           'StopTimeMonitor',
           'is_gpu_available',
           'is_ascend_available',
           'export_ms_model']
