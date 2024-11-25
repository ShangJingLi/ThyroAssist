import os
import subprocess
import time
import numpy as np
import mindspore
from mindspore import nn
from mindspore import Model

from src.deep_learning import MixedLoss
from src.deep_learning.dataloader import create_segmentation_dataset_at_numpy, download_and_unzip_segmentation_datasets
import mindspore.context as context
from mindspore.train.callback import LossMonitor
from src.deep_learning.networks import NestedUNet
from src.deep_learning.utils import get_time
from src.deep_learning.configuration import NestedUNetConfig

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
USE_ORANGE_PI = False
mindspore.set_seed(1)
config = NestedUNetConfig()
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


def trainer(epoch=config.train_epoch, batch_size=config.train_batch_size, lr=config.lr):
    if USE_ORANGE_PI:
        os.system('sudo npu-smi set -t pwm-duty-ratio -d 100')
    net = NestedUNet(n_channels=3, n_classes=2)
    if not os.path.exists('datasets_as_numpy'):
        download_and_unzip_segmentation_datasets()
    else:
        pass

    train_images = np.load(os.path.join("datasets_as_numpy", "train_images.npy"))
    train_masks = np.load(os.path.join("datasets_as_numpy", "train_masks.npy"))
    train_dataset = create_segmentation_dataset_at_numpy(train_images, train_masks,
                                                         img_size=config.image_size, mask_size=config.mask_size,
                                                         batch_size=batch_size, num_classes=2,
                                                         is_train=True, augment=False)
    loss_function = MixedLoss()
    optimizer = nn.Adam(params=net.trainable_params(), learning_rate=lr, weight_decay=0.0,
                        loss_scale=config.loss_scale)

    loss_scale_manager = mindspore.train.loss_scale_manager.FixedLossScaleManager(128, False)

    model = Model(net, loss_fn=loss_function, loss_scale_manager=loss_scale_manager,
                  optimizer=optimizer, metrics={"Dice系数": nn.Dice()}, amp_level='O0')  # nn.Accuracy
    print("============ Starting Training ============")
    start_time = time.time()
    model.train(epoch, train_dataset, callbacks=[LossMonitor(1)],
                dataset_sink_mode=False)
    current_directory = os.getcwd()
    target_directory = os.path.join(current_directory, 'segmentation_checkpoints')
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
