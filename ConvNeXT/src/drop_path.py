import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import Div as div_
from mindspore.numpy import empty_like
from mindspore import ops as P


def drop_path(x, drop_prob: float = 0, training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_ = empty_like(shape)

    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor = div_(random_tensor, keep_prob)
    return x * random_tensor


class DropPath(nn.Cell):
    """
    When drop_ When Path > 0, the multi branch structure will be inactivated randomly.

    Args:
        drop_prob(float): Random inactivation rate.
        scale_by_keep(bool): Default True.
    """
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def construct(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class Identity(nn.Cell):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def construct(self, input: Tensor) -> Tensor:
        return input