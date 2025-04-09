import os
import subprocess
import time
import numpy as np
import mindspore
from mindspore import nn
from mindspore import Model
from mindspore import context
from mindspore.train.callback import LossMonitor
import onnxruntime  # pylint: disable=unused-import 规避动态链接库异常
from thyassist.machine_learning.loss import MultiCrossEntropyWithLogits
from thyassist.machine_learning.dataloader import create_segmentation_dataset_at_numpy, download_and_unzip_segmentation_datasets
from thyassist.machine_learning.networks import NestedUNet
from thyassist.machine_learning.utils import get_time
from thyassist.machine_learning.configuration import NestedUNetConfig
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
mindspore.set_seed(1)
config = NestedUNetConfig()
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


def trainer(epoch=config.train_epoch, batch_size=config.train_batch_size, lr=config.lr):
    if USE_ORANGE_PI:
        os.system('sudo npu-smi set -t pwm-duty-ratio -d 100')
    if not os.path.exists(os.path.join(download_dir, 'datasets_as_numpy')):
        download_and_unzip_segmentation_datasets()
    else:
        pass

    train_images = np.load(os.path.join(download_dir, "datasets_as_numpy", "train_images.npy"))
    train_masks = np.load(os.path.join(download_dir, "datasets_as_numpy", "train_masks.npy"))
    train_dataset = create_segmentation_dataset_at_numpy(train_images, train_masks,
                                                         img_size=config.image_size, mask_size=config.mask_size,
                                                         batch_size=batch_size, num_classes=2,
                                                         is_train=True, augment=False)
    net = NestedUNet(n_channels=3, n_classes=2, is_train=True)
    loss_function = MultiCrossEntropyWithLogits()
    optimizer = nn.Adam(params=net.trainable_params(), learning_rate=lr, weight_decay=0.0,
                        loss_scale=config.loss_scale)

    loss_scale_manager = mindspore.train.loss_scale_manager.FixedLossScaleManager(128, False)
    early_stop = mindspore.EarlyStopping(monitor='eval_loss', min_delta=0, patience=50,
                                         verbose=True, mode='auto', restore_best_weights=True)
    model = Model(net, loss_fn=loss_function, loss_scale_manager=loss_scale_manager,
                  optimizer=optimizer, metrics={"Dice系数": nn.Dice()}, amp_level='O0')  # nn.Accuracy
    print("============ Starting Training ============")
    start_time = time.time()
    model.train(epoch, train_dataset, callbacks=[LossMonitor(1), early_stop],
                dataset_sink_mode=True)
    target_directory = os.path.join(download_dir, 'nested_unet_checkpoints')
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    else:
        pass
    mindspore.save_checkpoint(model.train_network, os.path.join(target_directory, 'nested_unet_checkpoints.ckpt'))
    end_time = time.time()
    print("训练时长：", get_time(start_time, end_time))
    print("============ End Training ============")
    if USE_ORANGE_PI:
        os.system('sudo npu-smi set -t pwm-duty-ratio -d 30')

trainer()
