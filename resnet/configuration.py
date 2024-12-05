import os


class ResNetConfig:
    def __init__(self):
        self.net_name = 'resnet50'
        self.num_class = 5
        self.train_images_path = os.path.join('datasets', 'train', 'train_images.npy')
        self.train_labels_path = os.path.join('datasets', 'train', 'train_labels.npy')
        self.val_images_path = os.path.join('datasets', 'val', 'val_images.npy')
        self.val_labels_path = os.path.join('datasets', 'val', 'val_labels.npy')
        self.infer_path = os.path.join('datasets', 'infer')
        self.output_dir = 'checkpoints'
        self.epoch_size = 10
        self.batch_size = 16
        self.pre_train = False
        self.ckpt_path = 'checkpoints/flowers_classification_ckpt.ckpt'
        self.warmup_epochs = 0
        self.lr_decay_mode = 'poly'
        self.lr_init = 0.01
        self.lr_end = 0.0001
        self.lr_max = 0.1
        self.weight_decay = 0.0001
        self.loss_scale = 1024.0
        self.boost_mode = 'O0'
        self.run_eval = True
        self.save_ckeckpoint = True
        self.dataset_sink_mode =  False

__all__ = ['ResNetConfig']
