"""

"""
from mindspore import nn
import numpy as np


def my_lr(start_lr, steps_all, decay, decay_step):
    """

    :param start_lr:
    :param steps_all:
    :param decay:
    :param decay_step:
    :return:
    """
    lrs = []
    lr_cur = start_lr
    for i in range(steps_all):
        lrs.append(lr_cur)
        if divmod(i+1, decay_step)[1] == 0:
            lr_cur = lr_cur * decay

    return lrs


def sigmoid(x):
    y = 1.0 / (1.0 + np.exp(-x))
    return y


class BinaryAcc(nn.Metric):
    """
    计算误比特率BER
    """
    def __init__(self):
        super(BinaryAcc, self).__init__()
        self.clear()

    def clear(self):
        self.data_num = 0
        self.bit_error_sum = 0

    def update(self, *inputs):
        y_pred = inputs[0].asnumpy()
        y_true = inputs[1].asnumpy()
        sigmoid_res = sigmoid(-y_pred)
        filt = np.where(sigmoid_res >= 0.5, 0, 1)
        pos = np.where(filt == y_true, 0, 1)
        self.bit_error_sum += np.sum(pos)
        (batch_size, n) = y_true.shape
        self.data_num += batch_size * n

    def eval(self):
        return self.bit_error_sum / self.data_num

# class MultiLoss(nn.LossBase):
#     """softmax and binary cross entropy loss function"""
#     def __init__(self):
#         super(MultiLoss, self).__init__()
#         self.bce = nn.SoftmaxCrossEntropyWithLogits()
#
#     def construct(self, base, target1, target2):
#         base1 = base[0]
#         base2 = base[1]
#         x1 = self.bceloss(base1, target1)
#         x2 = self.bceloss(base2, target2)
#         # x1 = self.bceloss(base1, target1.astype(mindspore.float16))
#         # x2 = self.bceloss(base2, target2.astype(mindspore.float16))
#         return self.get_loss(x1) + self.get_loss(x2)
