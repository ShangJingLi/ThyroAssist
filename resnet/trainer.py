# 若未下载数据集，请在运行trainer.py前下载数据集
import os
import warnings

import numpy as np
import mindspore
from mindspore import context
from mindspore import nn
from mindspore import Model
from mindspore.train.callback import LossMonitor, TimeMonitor
from resnet.configuration import ResNetConfig
from resnet.dataloader import create_dataset_with_numpy
from resnet.ResNet import resnet50
from resnet.functions import init_lr, init_group_params
from resnet.download_and_unzip_datasets import download_and_unzip
"""Train ResNet"""

mindspore.set_seed(1)
try:
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', save_graphs=False)
    print("采用NPU环境进行模型训练")
except:
    try:
        context.set_context(mode=context.GRAPH_MODE, device_target='GPU', save_graphs=False)
        print("当前系统不存在NPU环境，选择GPU环境执行训练")
    except:
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU", save_graphs=False)
        warnings.warn("NPU和GPU环境均不可用，回退到CPU环境进行模型训练,性能可能受限")


directory_name = 'datasets'
directory_path = os.path.join('', directory_name)

# 检查路径是否存在且是一个目录
if os.path.exists(directory_path) and os.path.isdir(directory_path):
    pass
else:
    print(f"目录 {directory_name} 不存在, 执行下载数据集。")
    download_and_unzip()


config = ResNetConfig()

net = resnet50(class_num=config.num_class)

train_images = np.load(config.train_images_path)
train_labels = np.load(config.train_labels_path)
ds_train = create_dataset_with_numpy(train_images, train_labels,
                                     batch_size=config.batch_size, is_train=True)
step_size = ds_train.get_dataset_size()


lr = mindspore.Tensor(init_lr(step_size=step_size))
group_params = init_group_params(net)
opt = nn.Adam(group_params, learning_rate=0.0001, loss_scale=config.loss_scale)


def init_loss_scale():
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=False, reduction='mean')
    return loss


loss = init_loss_scale()
loss_scale = mindspore.FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)

metrics = {'acc': nn.Accuracy()}
model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics=metrics,
              amp_level='O0', boost_level=config.boost_mode, keep_batchnorm_fp32=False, eval_network=None)


class LossCallBack(LossMonitor):
    def __init__(self, has_trained_epoch=0, per_print_times=5):
        super(LossCallBack, self).__init__()
        self.has_trained_epoch = has_trained_epoch
        self._per_print_times = per_print_times

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], mindspore.Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, mindspore.Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError('epoch: {} step: {}. Invalid loss, terminating training.'.format(
                cb_params.cur_epoch_num, cur_step_in_epoch))
        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            print("epoch: %s step: %s, loss is %s" % (cb_params.cur_epoch_num + int(self.has_trained_epoch),
                                                      cur_step_in_epoch, loss), flush=True)


time_cb = TimeMonitor(data_size=step_size)
loss_cb = LossCallBack(has_trained_epoch=0)

cb = [time_cb, loss_cb]
ckpt_save_dir = config.output_dir


print('========================== Starting Training ==========================')
model.train(config.epoch_size, ds_train, callbacks=cb,
            sink_size=ds_train.get_dataset_size(), dataset_sink_mode=config.dataset_sink_mode)

dir_name = "checkpoints"
current_directory = os.getcwd()
full_path = os.path.join(current_directory, dir_name)
if not os.path.exists(full_path):
    # 如果不存在，则创建目录
    os.makedirs(full_path)
    print(f"目录 {dir_name} 已创建。")

mindspore.save_checkpoint(model.train_network, 'checkpoints/flowers_classification_ckpt')
print(f"模型参数已成功保存于目录 ./{dir_name} 下")
print('========================== Training Ended ==========================')
