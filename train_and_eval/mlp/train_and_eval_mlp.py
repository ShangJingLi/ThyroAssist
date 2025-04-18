"""多层感知机训练脚本"""
import os
import subprocess
import mindspore
from mindspore import nn, context
from mindspore.train import LossMonitor
from thyassist.machine_learning.dataloader import create_mlp_dataset
from thyassist.machine_learning.configuration import MlpModelConfig
from thyassist.machine_learning.networks import CellSortMlp
from thyassist.machine_learning.dataloader import download_and_unzip_mlp_datasets
from launcher import get_project_root

#
#                       _oo0oo_
#                      o8888888o
#                      88" . "88
#                      (| -_- |)
#                      0\  =  /0
#                    ___/`---'\___
#                  .' \\|     |# '.
#                 / \\|||  :  |||# \
#                / _||||| -:- |||||- \
#               |   | \\\  -  #/ |   |
#               | \_|  ''\---/''  |_/ |
#               \  .-\__  '-'  ___/-. /
#             ___'. .'  /--.--\  `. .'___
#          ."" '<  `.___\_<|>_/___.' >' "".
#         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#         \  \ `_.   \_ __\ /__ _/   .-` /  /
#     =====`-.____`.___ \_____/___.-`___.-'=====
#                       `=---='
#
#
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#               佛祖保佑         永无BUG
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

if not os.path.exists(os.path.join(download_dir, 'mlp_datasets')):
    download_and_unzip_mlp_datasets()
else:
    pass

cancer_data_path = os.path.join(download_dir, "mlp_datasets", "cancer_cell.csv")
not_caner_data_path = os.path.join(download_dir, "mlp_datasets", "not_cancer_cell.csv")

train_dataset, eval_dataset = create_mlp_dataset(cancer_data_path,
                                                 not_caner_data_path)
loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=False, reduction='mean')
net = CellSortMlp()
opt = nn.Adam(net.trainable_params(), learning_rate=config.learning_rate)
model = mindspore.Model(net, loss_fn, opt, metrics={"准确率": nn.Accuracy()})


def train_and_eval():
    print("============ Starting Training ============")
    if USE_ORANGE_PI:
        os.system('sudo npu-smi set -t pwm-duty-ratio -d 100')
    model.train(epoch=10, train_dataset=train_dataset,
                callbacks=LossMonitor(per_print_times=1),
                dataset_sink_mode=config.dataset_sink_mode)
    metrics_result = model.eval(eval_dataset)
    print(metrics_result)

    target_directory = os.path.join(download_dir, "mlp_checkpoints")
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    else:
        pass
    if metrics_result['准确率'] == 1.0:
        print("精度达到100%，模型参数保存。")
        mindspore.save_checkpoint(model.train_network, os.path.join(target_directory,  "mlp_model_checkpoints.ckpt"))
    print("============== End Training ==============")
    if USE_ORANGE_PI:
        os.system('sudo npu-smi set -t pwm-duty-ratio -d 30')


train_and_eval()
