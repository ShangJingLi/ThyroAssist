import os
import subprocess
import time
import matplotlib.pyplot as plt
import numpy as np
import cv2
import mindspore
from mindspore import nn, context
from mindspore import Model, Tensor
from thyassist.machine_learning.dataloader import (create_segmentation_dataset_at_numpy,
                                                   download_and_unzip_nested_unet_checkpoints,
                                                   download_and_unzip_segmentation_datasets)
from thyassist.machine_learning.networks import NestedUNet
from thyassist.machine_learning.utils import get_time
from thyassist.machine_learning.configuration import NestedUNetConfig
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

config = NestedUNetConfig()
val_images = np.load(os.path.join(download_dir, "datasets_as_numpy", "val_images.npy"))
val_masks = np.load(os.path.join(download_dir, "datasets_as_numpy", "val_masks.npy"))

net = NestedUNet(n_channels=3, n_classes=2, is_train=False)
ckpt_directory = os.path.join(download_dir, 'nested_unet_checkpoints')
if not os.path.exists(ckpt_directory):
    download_and_unzip_nested_unet_checkpoints()
else:
    pass
ckpt_file = os.path.join(ckpt_directory, 'nested_unet_checkpoints.ckpt')
params = mindspore.load_checkpoint(ckpt_file)
mindspore.load_param_into_net(net, params)

loss_function = nn.DiceLoss()
optimizer = nn.Adam(params=net.trainable_params(), learning_rate=config.lr, weight_decay=0.00001,
                    loss_scale=config.loss_scale)

loss_scale_manager = mindspore.train.loss_scale_manager.FixedLossScaleManager(128, False)

model = Model(net, loss_fn=loss_function, loss_scale_manager=loss_scale_manager,
              optimizer=optimizer, metrics={"Dice系数": nn.Dice()}, amp_level='O0')  # nn.Accuracy


def eval_nested_unet():
    if USE_ORANGE_PI:
        os.system('sudo npu-smi set -t pwm-duty-ratio -d 100')
    if not os.path.exists(os.path.join(download_dir, 'datasets_as_numpy')):
        download_and_unzip_segmentation_datasets()
    else:
        pass

    val_dataset = create_segmentation_dataset_at_numpy(val_images, val_masks,
                                                       img_size=config.image_size, mask_size=config.mask_size,
                                                       batch_size=config.eval_batch_size, num_classes=2,
                                                       is_train=False, augment=False)

    print("============ Starting Evaluation ============")
    start_time = time.time()
    dice = model.eval(val_dataset, None, False)
    print(dice)
    end_time = time.time()
    print("评估时长：", get_time(start_time, end_time))
    print("============ End Evaluation ============")



def infer_nested_unet():
    if USE_ORANGE_PI:
        os.system('sudo npu-smi set -t pwm-duty-ratio -d 100')

    start_time = time.time()
    target_directory = os.path.join(download_dir, 'figures')
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    else:
        pass

    resized_images = np.zeros(shape=(val_images.shape[0], config.image_size[0],
                                     config.image_size[1], 3), dtype=np.uint8)

    # 遍历每个图像并调整大小
    for i in range(val_images.shape[0]):
        if len(val_images.shape) == 4:
            resized_images[i] = cv2.resize(val_images[i], dsize=config.image_size, interpolation=cv2.INTER_AREA)
        else:
            resized_images[i] = cv2.resize(cv2.cvtColor(val_images[i], cv2.COLOR_GRAY2BGR),
                                           dsize=config.image_size, interpolation=cv2.INTER_AREA)

    for i in range(10):
        origin_infer_data = resized_images[i]
        origin_mask = val_masks[i]

        infer_data = np.copy(origin_infer_data)
        infer_data = np.expand_dims(((infer_data.astype(np.float32)) / 127.5 - 1).transpose(2, 0, 1), axis=0)

        output = model.predict(Tensor(infer_data, dtype=mindspore.float32))
        output_as_numpy = np.argmax(output.asnumpy(), axis=1)
        output_as_numpy = output_as_numpy.reshape(256, 256)

        fig = plt.figure(figsize=(20, 10))
        plt.subplot(131)
        plt.imshow(origin_infer_data)
        plt.subplot(132)
        plt.imshow(origin_mask, cmap='gray')
        plt.subplot(133)
        plt.imshow(output_as_numpy, cmap='gray')
        fig.savefig(os.path.join(target_directory, f"{i}.jpg"))
    end_time = time.time()
    print("推理时长：", get_time(start_time, end_time))

    if USE_ORANGE_PI:
        os.system('sudo npu-smi set -t pwm-duty-ratio -d 30')


eval_nested_unet()
infer_nested_unet()
