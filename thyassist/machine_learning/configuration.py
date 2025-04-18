"""神经网络配置"""

class MlpModelConfig:
    """细胞分类多层感知机模型配置"""
    def __init__(self):
        self.num_features = 17
        self.num_classes = 2
        self.learning_rate = 0.001
        self.train_batch_size = 16
        self.eval_batch_size = 1
        self.dataset_sink_mode = False


class NestedUNetConfig:
    """NestedUNet模型配置"""
    def __init__(self):
        self.lr = 0.00001
        self.train_epoch = 50
        self.distribute_epochs = 1600
        self.train_batch_size = 16
        self.eval_batch_size = 1
        self.num_classes = 2
        self.weight_decay = 0.0001
        self.loss_scale = 1024
        self.resume = None
        self.image_size = (256, 256)
        self.mask_size = (256, 256)
