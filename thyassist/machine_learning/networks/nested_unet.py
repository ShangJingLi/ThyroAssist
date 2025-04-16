"""UNet++网络"""
from mindspore import nn, ops


def conv_bn_relu(in_channel, out_channel, use_bn=True, kernel_size=3, stride=1, pad_mode="same", activation='relu'):
    output = [nn.Conv2d(in_channel, out_channel, kernel_size, stride, pad_mode=pad_mode,
                        weight_init="normal", bias_init="zeros")]
    if use_bn:
        output.append(nn.BatchNorm2d(out_channel))
    if activation:
        output.append(nn.get_activation(activation))
    return nn.SequentialCell(output)


class ChannelAttention(nn.Cell):
    """
    Channel Attention module.
    """

    def __init__(self, in_planes, k, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=k, stride=k)
        self.max_pool = nn.MaxPool2d(kernel_size=k, stride=k)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias_init="zeros")
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias_init="zeros")
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class UnetConv2d(nn.Cell):
    """
    Convolution block in Unet, usually double conv.
    """

    def __init__(self, in_channel, out_channel, use_bn=True, num_layer=2, kernel_size=3, stride=1, padding='same'):
        super(UnetConv2d, self).__init__()
        self.num_layer = num_layer
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channel = in_channel
        self.out_channel = out_channel

        convs = []
        for _ in range(num_layer):
            convs.append(conv_bn_relu(in_channel, out_channel, use_bn, kernel_size, stride, padding, "relu"))
            in_channel = out_channel

        self.convs = nn.SequentialCell(convs)

    def construct(self, inputs):
        x = self.convs(inputs)
        return x


class UnetUp(nn.Cell):
    """
    Upsampling high_feature with factor=2 and concat with low feature
    """

    def __init__(self, in_channel, out_channel, use_deconv, k, n_concat=2):
        super(UnetUp, self).__init__()
        self.conv = UnetConv2d(in_channel + (n_concat - 2) * out_channel, out_channel, False)
        self.concat = ops.Concat(axis=1)
        self.use_deconv = use_deconv
        self.channel_attention = ChannelAttention(out_channel, k=k)  # 通道注意力模块
        if use_deconv:
            self.up_conv = nn.Conv2dTranspose(in_channel, out_channel, kernel_size=2, stride=2, pad_mode="same",
                                              weight_init="normal", bias_init="zeros")
        else:
            self.up_conv = nn.Conv2d(in_channel, out_channel, 1, weight_init="normal", bias_init="zeros")

    def construct(self, high_feature, *low_feature):
        if self.use_deconv:
            output = self.up_conv(high_feature)
        else:
            _, _, h, w = ops.shape(high_feature)
            output = ops.ResizeBilinearV2()(high_feature, (h * 2, w * 2))
            output = self.up_conv(output)
        for feature in low_feature:
            output = self.concat((output, feature))
        output = self.conv(output)
        output = output * self.channel_attention(output)  # 应用通道注意力
        return output


class NestedUNet(nn.Cell):
    """
    Nested unet with channel attention.
    """

    def __init__(self, is_train, n_channels=3, n_classes=2, feature_scale=2, use_deconv=True, use_bn=True):
        super(NestedUNet, self).__init__()
        self.in_channel = n_channels
        self.n_class = n_classes
        self.feature_scale = feature_scale
        self.use_deconv = use_deconv
        self.use_bn = use_bn
        self.use_ds = is_train

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # Down Sample
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same")
        self.conv00 = UnetConv2d(self.in_channel, filters[0], self.use_bn)
        self.conv10 = UnetConv2d(filters[0], filters[1], self.use_bn)
        self.conv20 = UnetConv2d(filters[1], filters[2], self.use_bn)
        self.conv30 = UnetConv2d(filters[2], filters[3], self.use_bn)
        self.conv40 = UnetConv2d(filters[3], filters[4], self.use_bn)

        # Up Sample
        self.up_concat01 = UnetUp(filters[1], filters[0], self.use_deconv, k=256, n_concat=2)
        self.up_concat11 = UnetUp(filters[2], filters[1], self.use_deconv, k=128, n_concat=2)
        self.up_concat21 = UnetUp(filters[3], filters[2], self.use_deconv, k=64, n_concat=2)
        self.up_concat31 = UnetUp(filters[4], filters[3], self.use_deconv, k=32, n_concat=2)

        self.up_concat02 = UnetUp(filters[1], filters[0], self.use_deconv, k=256, n_concat=3)
        self.up_concat12 = UnetUp(filters[2], filters[1], self.use_deconv, k=128, n_concat=3)
        self.up_concat22 = UnetUp(filters[3], filters[2], self.use_deconv, k=64, n_concat=3)

        self.up_concat03 = UnetUp(filters[1], filters[0], self.use_deconv, k=256, n_concat=4)
        self.up_concat13 = UnetUp(filters[2], filters[1], self.use_deconv, k=128, n_concat=4)

        self.up_concat04 = UnetUp(filters[1], filters[0], self.use_deconv, k=256, n_concat=5)

        # Finale Convolution
        self.final1 = nn.Conv2d(filters[0], n_classes, 1, weight_init="normal", bias_init="zeros")
        self.final2 = nn.Conv2d(filters[0], n_classes, 1, weight_init="normal", bias_init="zeros")
        self.final3 = nn.Conv2d(filters[0], n_classes, 1, weight_init="normal", bias_init="zeros")
        self.final4 = nn.Conv2d(filters[0], n_classes, 1, weight_init="normal", bias_init="zeros")
        self.stack = ops.Stack(axis=0)

    def construct(self, inputs):
        x00 = self.conv00(inputs)  # channel = filters[0]
        x10 = self.conv10(self.maxpool(x00))  # channel = filters[1]
        x20 = self.conv20(self.maxpool(x10))  # channel = filters[2]
        x30 = self.conv30(self.maxpool(x20))  # channel = filters[3]
        x40 = self.conv40(self.maxpool(x30))  # channel = filters[4]

        x01 = self.up_concat01(x10, x00)  # channel = filters[0]
        x11 = self.up_concat11(x20, x10)  # channel = filters[1]
        x21 = self.up_concat21(x30, x20)  # channel = filters[2]
        x31 = self.up_concat31(x40, x30)  # channel = filters[3]

        x02 = self.up_concat02(x11, x00, x01)  # channel = filters[0]
        x12 = self.up_concat12(x21, x10, x11)  # channel = filters[1]
        x22 = self.up_concat22(x31, x20, x21)  # channel = filters[2]

        x03 = self.up_concat03(x12, x00, x01, x02)  # channel = filters[0]
        x13 = self.up_concat13(x22, x10, x11, x12)  # channel = filters[1]

        x04 = self.up_concat04(x13, x00, x01, x02, x03)  # channel = filters[0]

        final1 = self.final1(x01)
        final2 = self.final2(x02)
        final3 = self.final3(x03)
        final4 = self.final4(x04)

        if self.use_ds:
            final = self.stack((final1, final2, final3, final4))
            return final
        return final4


__all__ = ["NestedUNet"]
