import mindspore
from mindspore import nn, ops, Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore import _checkparam as validator

class MixedLoss(nn.LossBase):
    def __init__(self, weight=None, gamma=2.0, smooth=1e-5, reduction='mean', loss_weight=mindspore.Tensor([4, 1])):
        """Initialize FocalLoss."""
        super(MixedLoss, self).__init__(reduction=reduction)

        self.gamma = validator.check_value_type("gamma", gamma, [float])
        if weight is not None and not isinstance(weight, Tensor):
            raise TypeError(f"For '{self.cls_name}', the type of 'weight' must be a Tensor, "
                            f"but got {type(weight).__name__}.")
        if isinstance(weight, Tensor) and weight.ndim != 1:
            raise ValueError(f"For '{self.cls_name}', the dimension of 'weight' must be 1, but got {weight.ndim}.")
        self.weight = weight
        self.expand_dims = P.ExpandDims()
        self.gather_d = P.GatherD()
        self.squeeze = P.Squeeze(axis=1)
        self.tile = P.Tile()
        self.cast = P.Cast()
        self.dtype = P.DType()
        self.logsoftmax = nn.LogSoftmax(1)
        self.smooth = validator.check_positive_float(smooth, "smooth")
        self.reshape = P.Reshape()
        self.loss_weight = (loss_weight.astype(mindspore.float32) / ops.sum(loss_weight))

    def construct(self, logits, label):
        # dice部分
        if logits.dtype == mindspore.uint8:
            raise TypeError(f"For '{self.cls_name}', the dtype of 'logits' can not be uint8.")
        if label.dtype == mindspore.uint8:
            raise TypeError(f"For '{self.cls_name}', the dtype of 'labels' can not be uint8.")
        intersection = self.reduce_sum(self.mul(logits.view(-1), label.view(-1)))  # ops.ReduceSum类和Mul类
        unionset = self.reduce_sum(self.mul(logits.view(-1), logits.view(-1))) + \
                   self.reduce_sum(self.mul(label.view(-1), label.view(-1)))  # mul是逐元素相乘，

        single_dice_coeff = (2 * intersection) / (unionset + self.smooth)
        dice_loss = self.loss_weight[0] * (1 - single_dice_coeff)

        # focal部分
        labelss = ops.argmax(label, dim=1, keepdim=True)

        if logits.ndim > 2:  # 我们是这种情况
            # 把（2， 388， 388）的输出拉成两条长向量
            logits = logits.view(logits.shape[0], logits.shape[1], -1)
            labelss = labelss.view(labelss.shape[0], labelss.shape[1], -1)
        else:
            logits = self.expand_dims(logits, 2)
            labelss = self.expand_dims(labelss, 2)

        log_probability = self.logsoftmax(logits)  # 对logit做softmax操作并取对数(ln)

        if label.shape[1] == 1:
            log_probability = self.gather_d(log_probability, 1, self.cast(labelss, mindspore.int32))
            log_probability = self.squeeze(log_probability)

        probability = F.exp(log_probability)  # 取指数，又回去了, 单纯对logits两个特征图展开做了个softmax

        if self.weight is not None:  # 按照道理要让前景更高，前景应该是放在第1维（从0开始数），故该填[1, 4]
            # 生成一个跟probability同形的数组（两条），每条代表一个权重
            convert_weight = self.weight[None, :, None]
            convert_weight = self.tile(convert_weight, (labelss.shape[0], 1, labelss.shape[2]))
            if label.shape[1] == 1:  # 不关我们的事
                convert_weight = self.gather_d(convert_weight, 1, self.cast(labelss, mindspore.int32))
                convert_weight = self.squeeze(convert_weight)
            # 将权重交给log_softmax之后的两条
            log_probability = log_probability * convert_weight

        weight = F.pows(-1 * probability + 1.0, self.gamma)
        if label.shape[1] == 1:
            # 不关我们的事
            loss = self.loss_weight[1] * (-1 * weight * log_probability).mean(axis=1)
        else:
            loss = self.loss_weight[1] * (-1 * weight * labelss * log_probability).mean(axis=-1)
        return self.get_loss(loss) + dice_loss

__all__ = ['MixedLoss']
