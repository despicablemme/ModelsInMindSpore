from mindspore import nn
from mindspore import Parameter
import numpy as np
import mindspore.ops as P


class LayerNorm(nn.Cell):
    """
    LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self,
                 normalized_shape: int,
                 eps: float = 1e-6,
                 data_format: str = "channels_last"):
        super(LayerNorm, self).__init__()
        self.weight = Parameter(np.ones(normalized_shape))
        self.bias = Parameter(np.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def construct(self, x):
        if self.data_format == "channels_last":
            return P.LayerNorm(input_x=x, gamma=self.weight,
                               beta=self.bias, epsilon=self.eps)
        elif self.data_format == "channels_first":
            mean = x.mean(1, keepdim=True)     # 在channel维度求均值
            var = (x - mean).pow(2).mean(1, keepdim=True)  # 求方差
            x = (x - mean) / np.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
