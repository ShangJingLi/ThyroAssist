# 请在运行trainer后再运行evaler.py
import os
import warnings
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mindspore
from mindspore import nn
from mindspore import Model
from mindspore import context
from resnet.configuration import ResNetConfig
from resnet.dataloader import create_dataset_with_numpy
from resnet.ResNet import resnet50
from resnet.functions import init_group_params
"""Evaluate ResNet"""


mindspore.set_seed(1)
try:
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', save_graphs=False)
    print("采用NPU环境进行模型评估和推理")
except:
    try:
        context.set_context(mode=context.GRAPH_MODE, device_target='GPU', save_graphs=False)
        print("当前系统不存在NPU环境，选择GPU环境执行模型评估和推理")
    except:
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU", save_graphs=False)
        warnings.warn("NPU和GPU环境均不可用，回退到CPU环境进行模型评估和推理,性能可能受限")


config = ResNetConfig()
best_ckpt_path = config.ckpt_path
network = resnet50(class_num=5)
net = network
loss = nn.SoftmaxCrossEntropyWithLogits(sparse=False, reduction='mean')
loss_scale = mindspore.FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
metrics = {'acc': nn.Accuracy()}
group_params = init_group_params(net)
opt = nn.Adam(group_params, learning_rate=0.0002, loss_scale=config.loss_scale)


def evaler(ckpt_path=best_ckpt_path):
    param_dict = mindspore.load_checkpoint(ckpt_path)
    mindspore.load_param_into_net(net, param_dict)
    model = Model(net, loss_fn=loss, optimizer=opt, metrics=metrics, amp_level='O0')

    val_images = np.load(config.val_images_path)
    val_labels = np.load(config.val_labels_path)
    ds_val = create_dataset_with_numpy(val_images, val_labels, batch_size=1, is_train=False)

    acc = model.eval(ds_val, None, True)
    print(acc)


def visualize_model(image_path, label, ckpt_path=best_ckpt_path):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_image)

    image = np.array(image).astype(np.float32)
    image = image / 127.5 - 1

    image = np.transpose(image, (2, 0, 1))

    image = np.expand_dims(image, axis=0)

    net = network

    param_dict = mindspore.load_checkpoint(ckpt_path)
    mindspore.load_param_into_net(net, param_dict)
    model = Model(net)

    pre = model.predict(mindspore.Tensor(image))
    print(pre)
    result = np.argmax(pre.asnumpy())

    class_name = {0: 'daisy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers', 4: 'tulips'}
    color = 'green' if class_name[result] == label else 'red'

    plt.title(f'Predict: {class_name[result]}\nlabel:{label}', color=color)
    plt.axis('off')
    return result


evaler()

image1 = os.path.join(config.infer_path, 'daisy.jpg')
plt.figure(figsize=(15, 7))
plt.subplot(1, 5, 1)
visualize_model(image1, 'daisy')

image2 = os.path.join(config.infer_path, 'dandelion.jpg')
plt.subplot(1, 5, 2)
visualize_model(image2, 'dandelion')

image3 = os.path.join(config.infer_path, 'roses.jpg')
plt.subplot(1, 5, 3)
visualize_model(image3, 'roses')

image4 = os.path.join(config.infer_path, 'sunflowers.jpg')
plt.subplot(1, 5, 4)
visualize_model(image4, 'sunflowers')

image5 = os.path.join(config.infer_path, 'tulips.jpg')
plt.subplot(1, 5, 5)
visualize_model(image5, 'tulips')

plt.show()
