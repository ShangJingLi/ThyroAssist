import os
import subprocess
import numpy as np
import mindspore
from mindspore import nn, Tensor, context
from thyassist.machine_learning.configuration import MlpModelConfig
from thyassist.machine_learning.networks import CellSortMlp
from launcher import get_project_root


download_dir = get_project_root()
USE_ORANGE_PI = False
if os.name == 'nt':
    context.set_context(mode=context.GRAPH_MODE)
    mindspore.set_device(device_target='CPU')
else:
    try:
        if subprocess.run(['whoami'], capture_output=True, text=True, check=True).stdout.strip() == 'HwHiAiUser':
            context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O2"})
            mindspore.set_device(device_target='Ascend')
            USE_ORANGE_PI = True
        else:
            context.set_context(mode=context.GRAPH_MODE)
            mindspore.set_device(device_target='Ascend')
    except:
        try:
            context.set_context(mode=context.GRAPH_MODE)
            mindspore.set_device(device_target="GPU")
        except:
            context.set_context(mode=context.GRAPH_MODE)
            mindspore.set_device(device_target="CPU")

config = MlpModelConfig()
ckpt_file = os.path.join(download_dir, "mlp_checkpoints", "mlp_model_checkpoints.ckpt")
net = CellSortMlp()
params = mindspore.load_checkpoint(ckpt_file)
mindspore.load_param_into_net(net, params)
opt = nn.Adam(net.trainable_params(), learning_rate=0.001)
loaded_model = mindspore.Model(net, nn.SoftmaxCrossEntropyWithLogits(reduction='mean'), opt)


def infer():
    while True:
        features_list = []
        print("Please input 'Area':")
        area = float(input())
        features_list.append(area)
        print("Please input 'Mean':")
        mean = float(input())
        features_list.append(mean)
        print("Please input 'StdDev':")
        stddev = float(input())
        features_list.append(stddev)
        print("Please input 'Mode':")
        mode = float(input())
        features_list.append(mode)
        print("Please input 'Min':")
        Min = float(input())
        features_list.append(Min)
        print("Please input 'Max':")
        Max = float(input())
        features_list.append(Max)
        print("Please input 'Perim.':")
        perim = float(input())
        features_list.append(perim)
        print("Please input 'Circ.':")
        circ = float(input())
        features_list.append(circ)
        print("Please input 'Feret':")
        feret = float(input())
        features_list.append(feret)
        print("Please input 'Median':")
        median = float(input())
        features_list.append(median)
        print("Please input 'FeretX':")
        feretX= float(input())
        features_list.append(feretX)
        print("Please input 'FeretY':")
        feretY = float(input())
        features_list.append(feretY)
        print("Please input 'FeretAngle':")
        feretangle = float(input())
        features_list.append(feretangle)
        print("Please input 'MinFeret':")
        feretangle = float(input())
        features_list.append(feretangle)
        print("Please input 'AR':")
        ar = float(input())
        features_list.append(ar)
        print("Please input 'Round':")
        Round = float(input())
        features_list.append(Round)
        print("Please input 'Solidity':")
        solidity = float(input())
        features_list.append(solidity)

        mean_array = np.load(os.path.join("../../../train_and_eval/mlp_datasets", "mean.npy"))
        std_array = np.load(os.path.join("../../../train_and_eval/mlp_datasets", "std.npy"))

        for i in range(len(features_list)):
            features_list[i] = (features_list[i] - mean_array[i]) / std_array[i]

        features_tensor = Tensor(features_list)
        features_tensor = features_tensor.expand_dims(0)
        result = loaded_model.predict(features_tensor)

        if result[0][0] > result[0][1]:
            print("根据预测结果，该细胞为癌细胞")
        else:
            print("根据预测结果，该细胞不是癌细胞\n\n")

        print("是否继续输入，若是则键盘输入‘y’，否则按下任意键退出")
        char = input()
        if char.lower() == 'y':
            pass
        else:
            break


__all__ = ['infer']
