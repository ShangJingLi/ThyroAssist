import numpy as np
from configuration import ResNetConfig
"Functions"

config = ResNetConfig()


def _generate_poly_lr(lr_init, lr_end, lr_max, total_steps, warmup_steps):
    lr_each_step = []
    if warmup_steps != 0:
        inc_each_step = (float(lr_max) - float(lr_init)) / warmup_steps
    else:
        inc_each_step = 0
    for i in range(total_steps):
        if i < warmup_steps:
            lr = float(lr_init) + inc_each_step * float(i)
        else:
            base = (1.0 - (float(i) - float(warmup_steps)) / (float(total_steps) - float(warmup_steps)))
            lr = float(lr_max) * base * base
            if lr < 0.0:
                lr = 0.0
        lr_each_step.append(lr)
    return lr_each_step


def get_lr(lr_init, lr_end, lr_max, warmup_epochs, total_epochs, steps_per_epoch, lr_decay_mode):
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    lr_each_step = _generate_poly_lr(lr_init, lr_end, lr_max, total_steps, warmup_steps)
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    return lr_each_step


def init_lr(step_size):
    if config.net_name in ('resnet18', 'resnet50', 'resnet34', 'resnet152'):
        lr = get_lr(lr_init=config.lr_init, lr_end=config.lr_end, lr_max=config.lr_max,
                    warmup_epochs=config.warmup_epochs, total_epochs=config.epoch_size,
                    steps_per_epoch=step_size, lr_decay_mode=config.lr_decay_mode)
    else:
        raise NameError("The name of net is wrong!")
    return lr


def init_group_params(net):
    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': config.weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': net.trainable_params()}]

    return group_params
