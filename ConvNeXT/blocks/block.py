from mindspore import nn
from mindspore import Parameter

import numpy as np

from layer_norm import LayerNorm
from drop_path import DropPath, Identity


class Block(nn.Cell):
    """
    ConvNext Block
    """
    def __init__(self,
                 dim: int,
                 drop_path: float = 0.0,
                 layer_scale_init_value: float = 1e-6):
        super(Block, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, pad_mode="pad", padding=3, group=dim)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channel_last")
        self.pwconv1 = nn.Dense(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Dense(4 * dim, dim)
        self.gamma = Parameter(layer_scale_init_value * np.ones((dim,)),
                               requires_grad=True) if layer_scale_init_value > 0 else None
        # drop_path 当drop_path>0时将多分支结构随机失活，否则是个恒等映射
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity

    def construct(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (NCHW)->(NHWC)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permuta(0, 3, 1, 2)   # (N,H, W, C) -> (N, C, H, W)

        x = shortcut + self.drop_path(x)
        return x