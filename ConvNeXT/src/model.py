# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import mindspore.nn as nn
from mindspore import Parameter
from mindspore import ops as P
from mindspore.common import initializer as init
from mindspore import Tensor
import mindspore as ms

import numpy as np

from src.drop_path import DropPath, Identity
from src.model_utils.moxing_adapter import config


class LayerNorm(nn.Cell):
    """
    LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
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


class Block(nn.Cell):
    """
    ConvNext Block
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
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


class ConvNeXt(nn.Cell):
    """
    Args:
        in_channel(int): Number of input image channels. Default: 3
        num_classes(int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_channel=3,
                 num_classes=1000,
                 depths=None,          # [3, 3, 9, 3]
                 dims=None,            # [96, 192, 384, 768]
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 head_init_scale=1):
        super(ConvNeXt, self).__init__()
        if depths is None:
            depths = [3, 3, 9, 3]
        if dims is None:
            dims = [96, 192, 384, 768]
##################################################################################
        self.down_sample_layers = nn.CellList()
        stem = nn.SequentialCell(
            nn.Conv2d(in_channel, dims[0], kernel_size=4, stride=4),
            LayerNorm(normalized_shape=dims[0],
                      eps=1e-6,
                      data_format="channels_first"))
        self.down_sample_layers.append(stem)

        # 构建stage2-stage4的前三个down_sample
        for i in range(3):
            down_sample_layer = nn.SequentialCell(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.down_sample_layers.append(down_sample_layer)
##################################################################################
        # 存储每一个stage所构建的一系列block
        self.stages = nn.CellList()
        linspace = P.LinSpace()
        start = Tensor(0, ms.float32)
        ls = linspace(start, drop_path_rate, sum(depths))

        dp_rates = [x.item((0, )) for x in linspace(start, drop_path_rate, sum(depths))]   # 等差数列

        cur = 0
        for i in range(4):
            stage = nn.SequentialCell(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value)
                  for j in range(depths[i])])
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm([dims[-1], ], epsilon=1e-6)
        self.head = nn.Dense(dims[-1], num_classes)

        self.head_init_scale = 1.0

        if config.initialize_mode == "Trunc":
            self._init_weights()

        # self.head.weight.data *= head_init_scale
        # self.head.bias.data *= head_init_scale

    def _init_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(init.initializer(init.TruncatedNormal(),
                                                      cell.weight.shape,
                                                      cell.weight.dtype))
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(cell.weight.data * self.head_init_scale)
                cell.bias.set_data(cell.bias.data * self.head_init_scale)

    def construct_feature(self, x):
        for i in range(4):
            x = self.down_sample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))      # global average pooling, (N, C, H, W) -> (N, C)

    def construct(self, x):
        x = self.construct_feature(x)
        x = self.head(x)
        return x


def convnext_tiny(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model


def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    return model