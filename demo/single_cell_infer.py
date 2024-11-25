import os
import subprocess
import sys
import signal
import gradio as gr
import numpy as np
import pandas as pd
import mindspore
from mindspore import Tensor
from mindspore import context
from src.deep_learning.dataloader import download_and_unzip_best_mlp_checkpoints, download_and_unzip_mlp_datasets
from src.deep_learning.networks import CellSortMlp
# cancer前大于后，not_cancer后大于前


USE_ORANGE_PI = False
if os.name == 'nt':
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
else:
    try:
        if subprocess.run(['whoami'], capture_output=True, text=True).stdout.strip() == 'HwHiAiUser':
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

if not os.path.exists('best_mlp_checkpoints'):
    download_and_unzip_best_mlp_checkpoints()
else:
    pass

if not os.path.exists('mlp_datasets'):
    download_and_unzip_mlp_datasets()
else:
    pass

means = np.load(os.path.join("mlp_datasets", "mean.npy"))
stds = np.load(os.path.join("mlp_datasets", "std.npy"))

net = CellSortMlp()
params = mindspore.load_checkpoint(os.path.join("best_mlp_checkpoints", "best_mlp_model_checkpoints.ckpt"))
mindspore.load_param_into_net(net, params)

if USE_ORANGE_PI:
    os.system('sudo npu-smi set -t pwm-duty-ratio -d 100')
    net(Tensor(np.zeros(shape=(1, 17)).astype(np.float32)))

signal.signal(signal.SIGINT, on_terminate)
signal.signal(signal.SIGTERM, on_terminate)

# 定义输入组件
dataframe_input = gr.Dataframe(
    label="Input DataFrame",
    headers=["Area", "Mean", "StdDev", "Mode", "Min", "Max", "Perim", "Circ", "Feret", "Median",
             "FeretX", "FeretY", "FeretAngle", "MinFeret", "AR", "Round", "Solidity"],  # 自定义列名
    row_count=1,  # 默认行数
    col_count=17,   # 默认列数
)

# 定义输出组件
output = gr.Dataframe(label='预测结果', headers=["Prediction"], row_count=1)

def infer_single_cell(input_datas):
    results = []
    for i in range(input_datas.shape[0]):
        data = input_datas.iloc[i:i+1, :]
        flag = 0
        try:
            numpy_data = data.iloc[:, :17].values.astype(np.float32)
            flag = 1
        except:
            results.append(['!!input error!!'])  # 转numpy有问题就直接返回error
        if flag == 1:
            processed_numpy_data = (numpy_data - means) / stds
            tensor_data = mindspore.Tensor(processed_numpy_data.astype(np.float32))
            result = net(tensor_data)
            if result[0][0] > result[0][1]:
                results.append(['==cancer cell=='])
            else:
                results.append(['==not cancer cell=='])
    output_datas = pd.DataFrame(results, columns=["预测结果"])
    return output_datas

# 创建 Gradio 接口
iface = gr.Interface(
    fn=infer_single_cell,
    inputs=dataframe_input,
    outputs=output,
    title="单甲状腺上皮细胞特征检测器",
    description="通过手动输入特征，使用多层感知机预测单个细胞是否属于癌细胞。"
)

iface.launch(share=True)