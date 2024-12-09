import os
import subprocess
import time
import matplotlib.pyplot as plt
import numpy as np
import cv2
import mindspore
from mindspore import nn, context
from mindspore import Model, Tensor
from src.deep_learning.dataloader import (create_segmentation_dataset_at_numpy,
                                          download_and_unzip_segmentation_checkpoints,
                                          download_and_unzip_segmentation_datasets)
from src.deep_learning.networks import NestedUNet
from src.deep_learning.utils import get_time
from src.deep_learning.configuration import NestedUNetConfig


USE_ORANGE_PI = False
if os.name == 'nt':
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
else:
    try:
        if subprocess.run(['whoami'], capture_output=True, text=True, check=True).stdout.strip() == 'HwHiAiUser':
            context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', jit_config={"jit_level": "O2"})
            USE_ORANGE_PI = True
        else:
            context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
    except:
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False)

config = NestedUNetConfig()


def eval_and_infer(infer_graph_mode=False):
    if USE_ORANGE_PI:
        os.system('sudo npu-smi set -t pwm-duty-ratio -d 100')
    if not os.path.exists('datasets_as_numpy'):
        download_and_unzip_segmentation_datasets()
    else:
        pass

    val_images = np.load(os.path.join("datasets_as_numpy", "val_images.npy"))
    val_masks = np.load(os.path.join("datasets_as_numpy", "val_masks.npy"))
    if len(val_images.shape) == 3:
        n_channels = 1
    else:
        n_channels = 3

    net = NestedUNet(n_channels=n_channels, n_classes=2, is_train=False)
    current_directory = os.getcwd()
    target_directory = os.path.join(current_directory, 'segmentation_checkpoints')
    if not os.path.exists(target_directory):
        download_and_unzip_segmentation_checkpoints()
    else:
        pass
    ckpt_file = os.path.join(target_directory, 'nested_unet_checkpoints.ckpt')
    params = mindspore.load_checkpoint(ckpt_file)
    mindspore.load_param_into_net(net, params)
    val_dataset = create_segmentation_dataset_at_numpy(val_images, val_masks,
                                                       img_size=config.image_size, mask_size=config.mask_size,
                                                       batch_size=config.eval_batch_size, num_classes=2,
                                                       is_train=False, augment=False)
    loss_function = nn.DiceLoss()
    optimizer = nn.Adam(params=net.trainable_params(), learning_rate=config.lr, weight_decay=0.00001,
                        loss_scale=config.loss_scale)

    loss_scale_manager = mindspore.train.loss_scale_manager.FixedLossScaleManager(128, False)

    model = Model(net, loss_fn=loss_function, loss_scale_manager=loss_scale_manager,
                  optimizer=optimizer, metrics={"Dice系数": nn.Dice()}, amp_level='O0')  # nn.Accuracy


    print("============ Starting Evaluation ============")
    start_time = time.time()
    dice = model.eval(val_dataset, None, False)
    print(dice)
    end_time = time.time()
    print("评估时长：", get_time(start_time, end_time))

    if infer_graph_mode:
        start_time = time.time()

        transposed_val_images = np.transpose(val_images, (0, 3, 1, 2))
        transposed_val_images = (transposed_val_images.astype(np.float32)) / 127.5 - 1
        inputs = Tensor(transposed_val_images)

        mindspore.export(net, inputs, file_name='unet_graph', file_format='MINDIR')
        graph = mindspore.load('unet_graph.mindir')
        net = nn.GraphCell(graph)

        outputs = net(inputs)
        outputs = np.argmax(outputs.asnumpy(), axis=1)
        end_time = time.time()
        print("推理时长：", get_time(start_time, end_time))

        current_directory = os.getcwd()
        target_directory = os.path.join(current_directory, 'figures')
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)
        else:
            pass
        for i in range(10):
            fig = plt.figure(figsize=(20, 10))
            plt.subplot(131)
            plt.imshow(val_images[2*i+1])
            plt.subplot(132)
            plt.imshow(val_masks[2*i+1], cmap='gray')
            plt.subplot(133)
            plt.imshow(outputs[2*i+1], cmap='gray')
            fig.savefig(os.path.join(target_directory, f"{2*i+1}.jpg"))

    else:
        start_time = time.time()
        current_directory = os.getcwd()
        target_directory = os.path.join(current_directory, 'figures')
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)
        else:
            pass

        for i in range(10):
            if len(val_images.shape) == 3:
                resized_shape = (val_images.shape[0], 256, 256)
            else:
                resized_shape = (val_images.shape[0], 256, 256, 3)
            resized_images = np.zeros(shape=resized_shape, dtype=np.uint8)
            # 遍历每个图像并调整大小
            for j in range(val_images.shape[0]):
                resized_images[j] = cv2.resize(val_images[j], (256, 256), interpolation=cv2.INTER_AREA)

            origin_infer_data = resized_images[i]
            origin_mask = val_masks[i]

            infer_data = np.copy(origin_infer_data)
            if len(infer_data.shape) == 3:
                infer_data = np.expand_dims(((infer_data.astype(np.float32)) / 127.5 - 1).transpose(2, 0, 1), axis=0)
            else:
                infer_data = np.reshape(infer_data, (1, 1, infer_data.shape[0], infer_data.shape[1])) / 127.5 - 1

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
    print("============ End Evaluation ============")
    if USE_ORANGE_PI:
        os.system('sudo npu-smi set -t pwm-duty-ratio -d 30')

eval_and_infer()
