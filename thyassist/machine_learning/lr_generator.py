"""生成ResNet152训练学习率"""
import math
import numpy as np


def _generate_steps_lr(lr_init, lr_max, total_steps, warmup_steps, start_steps):
    """
    Applies three steps decay to generate learning rate array.

    Args:
       lr_init(float): init learning rate.
       lr_max(float): max learning rate.
       total_steps(int): all steps in training.
       warmup_steps(int): all steps in warmup epochs.

    Returns:
       np.array, learning rate array.
    """
    decay_epoch_index = [0.3 * total_steps, 0.6 * total_steps, 0.8 * total_steps]
    lr_each_step = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr = lr_init + (lr_max - lr_init) * i / warmup_steps
        else:
            if i < decay_epoch_index[0]:
                lr = lr_max
            elif i < decay_epoch_index[1]:
                lr = lr_max * 0.1
            elif i < decay_epoch_index[2]:
                lr = lr_max * 0.01
            else:
                lr = lr_max * 0.001
        lr_each_step.append(lr)
    return lr_each_step[start_steps:]


def _generate_step_lr(lr_max, total_steps, start_steps):
    """
    Applies three steps decay to generate learning rate array.

    Args:
       lr_init(float): init learning rate.
       lr_max(float): max learning rate.
       total_steps(int): all steps in training.
       warmup_steps(int): all steps in warmup epochs.

    Returns:
       np.array, learning rate array.
    """
    decay_epoch_index = [0.2 * total_steps, 0.5 * total_steps, 0.7 * total_steps, 0.9 * total_steps]
    lr_each_step = []
    for i in range(total_steps):
        if i < decay_epoch_index[0]:
            lr = lr_max
        elif i < decay_epoch_index[1]:
            lr = lr_max * 0.1
        elif i < decay_epoch_index[2]:
            lr = lr_max * 0.01
        elif i < decay_epoch_index[3]:
            lr = lr_max * 0.001
        else:
            lr = 0.00005
        lr_each_step.append(lr)
    return lr_each_step[start_steps:]


def _generate_poly_lr(lr_init, lr_max, total_steps, warmup_steps, start_steps):
    """
    Applies polynomial decay to generate learning rate array.

    Args:
       lr_init(float): init learning rate.
       lr_end(float): end learning rate
       lr_max(float): max learning rate.
       total_steps(int): all steps in training.
       warmup_steps(int): all steps in warmup epochs.

    Returns:
       np.array, learning rate array.
    """
    lr_each_step = []
    if warmup_steps != 0:
        inc_each_step = (float(lr_max) - float(lr_init)) / float(warmup_steps)
    else:
        inc_each_step = 0
    for i in range(total_steps):
        if i < warmup_steps:
            lr = float(lr_init) + inc_each_step * float(i)
        else:
            base = (1.0 - (float(i) - float(warmup_steps)) / (float(total_steps) - float(warmup_steps)))
            lr = float(lr_max) * base * base
            lr = max(lr, 1e-5)
        lr_each_step.append(lr)
    return lr_each_step[start_steps:]


def _generate_cosine_lr(lr_init, lr_max, total_steps, warmup_steps, start_steps):
    """
    Applies cosine decay to generate learning rate array.

    Args:
       lr_init(float): init learning rate.
       lr_end(float): end learning rate
       lr_max(float): max learning rate.
       total_steps(int): all steps in training.
       warmup_steps(int): all steps in warmup epochs.

    Returns:
       np.array, learning rate array.
    """
    decay_steps = total_steps - warmup_steps
    lr_each_step = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr_inc = (float(lr_max) - float(lr_init)) / float(warmup_steps)
            lr = float(lr_init) + lr_inc * (i + 1)
        else:
            linear_decay = (total_steps - i) / decay_steps
            cosine_decay = 0.5 * (1 + math.cos(math.pi * 2 * 0.47 * i / decay_steps))
            decayed = linear_decay * cosine_decay + 0.00001
            lr = lr_max * decayed
        lr_each_step.append(lr)
    return lr_each_step[start_steps:]


def _generate_liner_lr(lr_init, lr_end, lr_max, total_steps, warmup_steps, start_steps):
    """
    Applies liner decay to generate learning rate array.

    Args:
       lr_init(float): init learning rate.
       lr_end(float): end learning rate
       lr_max(float): max learning rate.
       total_steps(int): all steps in training.
       warmup_steps(int): all steps in warmup epochs.

    Returns:
       np.array, learning rate array.
    """
    lr_each_step = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr = lr_init + (lr_max - lr_init) * i / warmup_steps
        else:
            lr = lr_max - (lr_max - lr_end) * (i - warmup_steps) / (total_steps - warmup_steps)
        lr_each_step.append(lr)
    return lr_each_step[start_steps:]


def get_lr(lr_init, lr_end, lr_max, warmup_epochs, total_epochs, start_epoch, steps_per_epoch, lr_decay_mode):
    """
    generate learning rate array

    Args:
       lr_init(float): init learning rate
       lr_end(float): end learning rate
       lr_max(float): max learning rate
       warmup_epochs(int): number of warmup epochs
       total_epochs(int): total epoch of training
       steps_per_epoch(int): steps of one epoch
       lr_decay_mode(string): learning rate decay mode, including steps, poly, cosine or liner(default)

    Returns:
       np.array, learning rate array
    """
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    start_steps = steps_per_epoch * start_epoch

    if lr_decay_mode == 'steps':
        lr_each_step = _generate_steps_lr(lr_init, lr_max, total_steps, warmup_steps, start_steps)
    elif lr_decay_mode == 'step':
        lr_each_step = _generate_step_lr(lr_max, total_steps, start_steps)
    elif lr_decay_mode == 'poly':
        lr_each_step = _generate_poly_lr(lr_init, lr_max, total_steps, warmup_steps, start_steps)
    elif lr_decay_mode == 'cosine':
        lr_each_step = _generate_cosine_lr(lr_init, lr_max, total_steps, warmup_steps, start_steps)
    else:
        lr_each_step = _generate_liner_lr(lr_init, lr_end, lr_max, total_steps, warmup_steps, start_steps)

    lr_each_step = np.array(lr_each_step).astype(np.float32)
    return lr_each_step


def linear_warmup_lr(current_step, warmup_steps, base_lr, init_lr):
    lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
    lr = float(init_lr) + lr_inc * current_step
    return lr


def warmup_cosine_annealing_lr(lr, steps_per_epoch, warmup_epochs, max_epoch=120, global_step=0):
    """
    generate learning rate array with cosine

    Args:
       lr(float): base learning rate
       steps_per_epoch(int): steps size of one epoch
       warmup_epochs(int): number of warmup epochs
       max_epoch(int): total epochs of training
       global_step(int): the current start index of lr array
    Returns:
       np.array, learning rate array
    """
    base_lr = lr
    warmup_init_lr = 0
    total_steps = int(max_epoch * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    decay_steps = total_steps - warmup_steps

    lr_each_step = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr = linear_warmup_lr(i + 1, warmup_steps, base_lr, warmup_init_lr)
        else:
            linear_decay = (total_steps - i) / decay_steps
            cosine_decay = 0.5 * (1 + math.cos(math.pi * 2 * 0.47 * i / decay_steps))
            decayed = linear_decay * cosine_decay + 0.00001
            lr = base_lr * decayed
        lr_each_step.append(lr)

    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[global_step:]
    return learning_rate


def get_thor_lr(global_step, lr_init, decay, total_epochs, steps_per_epoch, decay_epochs=100):
    """get_model_lr"""
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    for i in range(total_steps):
        epoch = (i + 1) / steps_per_epoch
        base = (1.0 - float(epoch) / total_epochs) ** decay
        lr_local = lr_init * base
        if epoch >= decay_epochs:
            lr_local = lr_local * 0.5
        if epoch >= decay_epochs + 1:
            lr_local = lr_local * 0.5
        lr_each_step.append(lr_local)
    current_step = global_step
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[current_step:]
    return learning_rate


def get_thor_damping(global_step, damping_init, decay_rate, total_epochs, steps_per_epoch):
    """get_model_damping"""
    damping_each_step = []
    total_steps = steps_per_epoch * total_epochs
    for step in range(total_steps):
        epoch = (step + 1) / steps_per_epoch
        damping_here = damping_init * (decay_rate ** (epoch / 10))
        damping_each_step.append(damping_here)
    current_step = global_step
    damping_each_step = np.array(damping_each_step).astype(np.float32)
    damping_now = damping_each_step[current_step:]
    return damping_now
