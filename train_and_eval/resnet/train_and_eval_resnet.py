import os
import subprocess
import numpy as np
import mindspore
from mindspore import nn, context
import mindspore.log as logger
from thyassist.machine_learning.resnet_configuration import config
from thyassist.machine_learning.networks import resnet152
from thyassist.machine_learning import lr_generator
from thyassist.machine_learning.loss import CrossEntropySmooth
from thyassist.machine_learning.dataloader.load_resnet_data import (create_dataset_with_numpy,
                                                                    convert_to_numpy,
                                                                    generate_images_and_labels,
                                                                    )
from thyassist.machine_learning.dataloader.download_resnet_data import download_and_unzip_resnet_datasets
from launcher import get_project_root


download_dir = get_project_root()
USE_ORANGE_PI = False
mindspore.set_seed(1)
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


def init_lr(step_size):
    """init lr"""
    if config.optimizer == "Thor":
        from thyassist.machine_learning.lr_generator import get_thor_lr
        lr = get_thor_lr(config.start_epoch * step_size, config.lr_init, config.lr_decay, config.lr_end_epoch,
                         step_size, decay_epochs=39)
    else:
        if config.net_name in ("resnet18", "resnet34", "resnet50", "resnet152", "se-resnet50"):
            config.lr_max = config.lr_max / 8 * config.device_num
            lr = lr_generator.get_lr(lr_init=config.lr_init, lr_end=config.lr_end, lr_max=config.lr_max,
                        warmup_epochs=config.warmup_epochs, total_epochs=config.epoch_size,
                        start_epoch=config.start_epoch, steps_per_epoch=step_size, lr_decay_mode=config.lr_decay_mode)
        else:
            lr = lr_generator.warmup_cosine_annealing_lr(config.lr, step_size, config.warmup_epochs, config.epoch_size,
                                            config.start_epoch * step_size)
    return lr


def init_loss_scale():
    if config.dataset == "imagenet2012":
        if not config.use_label_smooth:
            config.label_smooth_factor = 0.0
        loss = CrossEntropySmooth(sparse=True, reduction="mean",
                                  smooth_factor=config.label_smooth_factor, num_classes=config.class_num)
    else:
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=False, reduction='mean')
    return loss


def set_ascend_max_device_memory():
    if mindspore.get_context("enable_ge") and mindspore.get_context("mode") == 0 and \
       hasattr(config, "max_device_memory"):
        logger.warning("When encountering a memory shortage situation, reduce the max_device_memory.")
        mindspore.set_context(max_device_memory=config.max_device_memory)


def create_datasets(images_path, method, padding=None, aim_size=(572, 572)):
    images_a, images_b = convert_to_numpy(images_path, method=method, padding=padding, aim_size=aim_size)
    train_images, train_labels, val_images, val_labels = generate_images_and_labels(images_a, images_b)
    return train_images, train_labels, val_images, val_labels



def train_net(train_images, train_labels, batch_size, method):
    """train net"""
    train_datasets = create_dataset_with_numpy(images=train_images, labels=train_labels, batch_size=batch_size, is_train=True)
    net = resnet152(class_num=2)

    step_size = train_datasets.get_dataset_size()
    lr = mindspore.Tensor(init_lr(step_size=step_size))
    # define opt
    opt = nn.Adam(params=net.trainable_params(), learning_rate=lr)
    loss = init_loss_scale()
    loss_scale = mindspore.FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    metrics = {"acc"}
    model = mindspore.Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics=metrics,
                            amp_level="O0", boost_level=config.boost_mode,
                            boost_config_dict={"grad_freeze": {"total_steps": config.epoch_size * step_size}})
    # train model
    print('========================== Starting Training ==========================')
    model.train(30, train_datasets, callbacks=mindspore.LossMonitor(1),
                sink_size=train_datasets.get_dataset_size(), dataset_sink_mode=True)
    print('========================== Training Ended ==========================')
    target_directory = os.path.join(download_dir, f'medical_resnet_checkpoints({method})')
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    else:
        pass
    mindspore.save_checkpoint(model.train_network, os.path.join(target_directory, 'medical_resnet_checkpoints.ckpt'))

def eval_net(val_images, val_labels, method):
    eval_datasets = create_dataset_with_numpy(val_images, val_labels, batch_size=1, is_train=False)
    net = resnet152()
    metrics = {"acc"}
    params = mindspore.load_checkpoint(os.path.join(download_dir, f"medical_resnet_checkpoints({method})", "medical_resnet_checkpoints.ckpt"))
    mindspore.load_param_into_net(net, params)
    loss = init_loss_scale()
    lr = 0.001
    # define opt
    opt = nn.Adam(params=net.trainable_params(), learning_rate=lr)
    model = mindspore.Model(net, loss_fn=loss, optimizer=opt,metrics=metrics, boost_level=config.boost_mode)

    acc = model.eval(eval_datasets, None, False)
    print(acc)


if __name__ == '__main__':
    if USE_ORANGE_PI:
        os.system('sudo npu-smi set -t pwm-duty-ratio -d 100')

    use_custom_datasets = False
    custom_datasets_path = None
    batch_size = 16

    method = "pad"
    if not use_custom_datasets:
        if method == "pad":
            if not (os.path.exists(os.path.join(download_dir, 'padding_datasets'))):
                download_and_unzip_resnet_datasets(method="pad")
            train_images = np.load(os.path.join(download_dir, "padding_datasets", "train_images.npy"))
            train_labels = np.load(os.path.join(download_dir, "padding_datasets", "train_labels.npy"))
            val_images = np.load(os.path.join(download_dir, "padding_datasets", "val_images.npy"))
            val_labels = np.load(os.path.join(download_dir, "padding_datasets", "val_labels.npy"))
        else:
            if not (os.path.exists(os.path.join(download_dir, 'crop_datasets'))):
                download_and_unzip_resnet_datasets(method="crop")
            train_images = np.load(os.path.join(download_dir, "crop_datasets", "train_images.npy"))
            train_labels = np.load(os.path.join(download_dir, "crop_datasets", "train_labels.npy"))
            val_images = np.load(os.path.join(download_dir, "crop_datasets", "val_images.npy"))
            val_labels = np.load(os.path.join(download_dir, "crp[_datasets", "val_labels.npy"))
        train_net(train_images, train_labels, batch_size=16, method=method)
        eval_net(val_images, val_labels, method=method)

    else:
        images_a, images_b = convert_to_numpy(images_path=custom_datasets_path, method=method)
        train_images, train_labels, val_images, val_labels = generate_images_and_labels(images_a,
                                                                                        images_b,
                                                                                        batch_size=batch_size)
        train_net(train_images, train_labels, batch_size=batch_size, method=method)
        eval_net(val_images, val_labels, method=method)
