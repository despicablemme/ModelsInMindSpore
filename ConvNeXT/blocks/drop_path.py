"""DropPath module."""

from mindspore import nn
from mindspore import Tensor
import mindspore.nn.probability.distribution as msd
from mindspore.numpy import empty_like
from mindspore.ops import Div as div_


class DropPath(nn.Cell):
    """

    """

    def __init__(self, drop_prob=0, training=False, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1 - self.drop_prob
        self.scale_by_keep = scale_by_keep
        self.probs = self.keep_prob
        self.bernoulli = msd.Bernoulli(probs=self.probs)
        self.training = training

    def construct(self, x):
        if self.drop_prob == 0 or not self.training:
            return x
        random_tensor = self.bernoulli.sample(empty_like((x.shape[0], ) + (1, ) * (x.ndim - 1)))
        if self.keep_prob > 0.0 and self.scale_by_keep:
            random_tensor = div_(random_tensor, self.keep_prob)
        return x * random_tensor


class Identity(nn.Cell):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def construct(self, input: Tensor) -> Tensor:
        return input