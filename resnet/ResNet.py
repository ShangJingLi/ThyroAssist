import math

from mindspore import nn, ops
from mindspore.common.initializer import HeUniform, HeNormal, initializer
"""ResNet"""


def _conv33(in_channel, out_channel, stride=1, res_base=False):
    weight_shape = (out_channel, in_channel, 3, 3)
    weight = initializer(HeNormal(mode='fan_out', nonlinearity='relu'), shape=weight_shape)
    if res_base:
        return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1,
                         pad_mode='pad', weight_init=weight)  #
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride,
                     padding=0, pad_mode='same', weight_init=weight)  #


def _conv77(in_channel, out_channel, stride=1, res_base=False):
    weight_shape = (out_channel, in_channel, 7, 7)
    weight = initializer(HeNormal(mode='fan_out', nonlinearity='relu'), shape=weight_shape)
    if res_base:
        return nn.Conv2d(in_channel, out_channel, kernel_size=7, stride=stride, padding=3,
                         pad_mode='pad', weight_init=weight)  #
    return nn.Conv2d(in_channel, out_channel, kernel_size=7, stride=stride, padding=0,
                     pad_mode='same', weight_init=weight)  #


def _bn(channel, res_base=False):
    if res_base:
        return nn.BatchNorm2d(channel, eps=1e-5, momentum=0.1,
                              gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9, gamma_init=1, beta_init=0,
                          moving_mean_init=0, moving_var_init=1)


def _conv11(in_channel, out_channel, stride=1, use_se=False, res_base=False):
    weight_shape = (out_channel, in_channel, 1, 1)
    weight = initializer(HeNormal(mode='fan_out', nonlinearity='relu'), shape=weight_shape)
    if res_base:
        return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride,
                         padding=0, pad_mode='pad', weight_init=weight)  #
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride,
                     padding=0, pad_mode='same', weight_init=weight)  #


def _fc(in_channel, out_channel):
    weight_shape = (out_channel, in_channel)
    weight = initializer(HeUniform(negative_slope=math.sqrt(5)), shape=weight_shape)
    return nn.Dense(in_channel, out_channel, has_bias=False, bias_init=0, weight_init=weight)  #

# class AttentionBlock(nn.Cell):
#     def __init__(self, in_channel, out_channel):
#         super(AttentionBlock, self).__init__()
#         self.Wq = nn.Dense(in_channel, )
#
#     def construct(self, x):
#         q =

class ResidualBlock(nn.Cell):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, use_se=False, se_block=False):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        self.use_se = use_se
        self.se_block = se_block
        channel = out_channel // self.expansion
        self.conv1 = _conv11(in_channel, channel, stride=1)
        self.bn1 = _bn(channel)

        self.conv2 = _conv33(channel, channel, stride=stride)
        self.bn2 = _bn(channel)

        self.conv3 = _conv11(channel, out_channel, stride=1)
        self.bn3 = _bn(out_channel)
        self.relu = nn.ReLU()

        self.down_sample = False

        if stride != 1 or in_channel != out_channel:
            self.down_sample = True
        self.down_sample_layer = None

        if self.down_sample:
            self.down_sample_layer = nn.SequentialCell([_conv11(in_channel, out_channel, stride), _bn(out_channel)])

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample:
            identity = self.down_sample_layer(identity)

        out = out + identity
        out = self.relu(out)

        return out


class ResNet(nn.Cell):
    def __init__(self, block, layer_nums, in_channels, out_channels, strides, num_classes, res_base=False):
        super(ResNet, self).__init__()

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("The length of layer_num, in_channels, out_channels list must be 4!")
        self.res_base = res_base

        self.conv1 = _conv77(3, 64, stride=2, res_base=self.res_base)
        self.bn1 = _bn(64, self.res_base)
        self.relu = nn.ReLU()

        if self.res_base:
            self.pad = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid')
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')

        self.layer1 = self._make_layer(block, layer_nums[0], in_channel=in_channels[0],
                                       out_channel=out_channels[0], stride=strides[0])
        self.layer2 = self._make_layer(block, layer_nums[1], in_channel=in_channels[1],
                                       out_channel=out_channels[1], stride=strides[1])
        self.layer3 = self._make_layer(block, layer_nums[2], in_channel=in_channels[2],
                                       out_channel=out_channels[2], stride=strides[2])
        self.layer4 = self._make_layer(block, layer_nums[3], in_channel=in_channels[3],
                                       out_channel=out_channels[3], stride=strides[3])

        self.mean = ops.ReduceMean(keep_dims=True)
        self.flatten = nn.Flatten()
        self.end_point = _fc(out_channels[3], 64)
        self.fc1 = _fc(64, 256)
        self.fc2 = _fc(256, 32)
        self.fc3 = _fc(32, num_classes)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights for cells."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(
                    initializer(HeNormal(mode='fan_out', nonlinearity='relu'),
                                cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(
                        initializer('zeros', cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(initializer('ones', cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(initializer('zeros', cell.beta.shape, cell.beta.dtype))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(
                    initializer(HeUniform(mode='fan_in', nonlinearity='sigmoid'),
                                cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.res_base:
            x = self.pad(x)

        c1 = self.maxpool(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        out = self.mean(c5, (2, 3))
        out = self.flatten(out)
        out = self.end_point(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

    @staticmethod
    def _make_layer(block, layer_num, in_channel, out_channel, stride):
        layers = []
        resnet_block = block(in_channel, out_channel, stride=stride)
        layers.append(resnet_block)
        for _ in range(1, layer_num):
            resnet_block = block(out_channel, out_channel, stride=1)
            layers.append(resnet_block)
        return nn.SequentialCell(layers)


def resnet50(class_num=5):
    return ResNet(ResidualBlock, layer_nums=[3, 4, 6, 3], in_channels=[64, 256, 512, 1024],
                  out_channels=[256, 512, 1024, 2048], strides=[1, 2, 2, 2], num_classes=class_num)
