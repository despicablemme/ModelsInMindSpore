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
"""ConvNext backbone."""
import mindspore.nn as nn
from mindspore import Parameter
from mindspore import ops as P
from mindspore.common import initializer as init
from mindspore import Tensor
import mindspore as ms

import numpy as np

from drop_path import DropPath, Identity
from mindvision.engine.class_factory import ClassFactory, ModuleType

__all__ = [
    "LayerNorm",
    "Block",
    "ConvNeXt",
    "ConvNeXtTiny",
    "ConvNeXtSmall",
    "ConvNeXtBase",
    "ConvNeXtLarge",
    "ConvNeXtxLarge"
]


class LayerNormCell(nn.Cell):
    """
    LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).

    Args:
        normalized_shape(int): Normalization along axis.
        eps(float): A value added to the denominator for numerical stability. Default: 1e-7.
        data_format(str):channels_last corresponds to inputs with shape (batch_size, height, width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width).

    Returns:
        normalized x

    Examples:
        >>> LayerNormCell(normalized_shape=96, eps=1e-6, data_format="channels_last")
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super(LayerNormCell, self).__init__()
        self.weight = Parameter(np.ones(normalized_shape))
        self.bias = Parameter(np.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def construct(self, x):
        if self.data_format == "channels_last":
            return P.LayerNorm(input_x=x, gamma=self.weight,
                               beta=self.bias, epsilon=self.eps)
        elif self.data_format == "channels_first":
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / np.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Cell):
    """
    ConvNext Block. There are two equivalent implementations:
    (1) DwConv -> layernorm(channel_first)->1*1 Conv —>GELU -> 1*1 Conv,all in (N, C, H, W);
    (2) DwConv -> Permute to (NHWC), layernorm(channels_last) -> Dense -> GELU -> Dense,
    permute back to (NCHW). We use (2).

    Args:
        dim(int):Number of input channels.
        drop_path(float): Stochastic depth rate. Default:0.0
        layer_scale_init_value(float): Init value for Layer Scale. Default:1e-6

    Returns:
        tensor

    Examples:
        >>> from mindvision.classification.models.backbones import Block
        >>> Block(dim=3, drop_path=0, layer_scale_init_value=1e-6)
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super(Block, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, pad_mode="pad", padding=3, group=dim)
        self.norm = LayerNormCell(dim, eps=1e-6, data_format="channel_last")
        self.pwconv1 = nn.Dense(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Dense(4 * dim, dim)
        self.gamma = Parameter(layer_scale_init_value * np.ones((dim,)),
                               requires_grad=True) if layer_scale_init_value > 0 else None
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
    ConvNext architecture.

    Args:
        in_channel(int): Number of input image channels. Default: 3
        num_classes(int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.

    Inputs:
        - **x**(Tensor) - Tensor of shape:(batch_size, Channel, Height, Weight)

    Outputs:
        Tensor, output of tensor.

    Supported Platforms:
        ''GPU'''

    Examples:
        >>> from mindvision.classification.models.backbones import ConvNext, Blocks, LayerNorm
        >>> import numpy as np
        >>> x = np.random.randn(32*3*224*224).reshape(32, 3, 224, 224)
        >>> net = ConvNeXt(in_channel=3, num_classes=10, depths=[3, 3, 9, 3], dims=[96, 192, 284, 768], drop_path_rate=0, layer_scale_init_value=1e-6, head_init_scale=1)
        >>> output = net(x)
        >>> print(output.shape)

    About ConvNext:

    Starting from resnet-50 or resnet-200, convnext successively draws lessons from
    the idea of swin transformer from the five perspectives of macro design, deep separable
    convolution (resnext), inverse bottleneck layer (mobilenet V2), large convolution
    core and detailed design, and then carries out training and evaluation on imagenet-1k,
    and finally obtains the core structure of convnext.

     Citation:

     .. code-block::

        @article{,
        title={A ConvNet for the 2020s},
        author={Zhuang, Liu. and Hanzi, Mao. and Chao-Yuan, Wu.},
        journal={},
        year={}
        }
    """

    def __init__(self, in_channel=3,
                 num_classes=1000,
                 depths=None,
                 dims=None,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 head_init_scale=1.0):
        super(ConvNeXt, self).__init__()

        if depths is None:
            depths = [3, 3, 9, 3]
        if dims is None:
            dims = [96, 192, 384, 768]

        self.downsample_layers = nn.CellList()
        stem = nn.SequentialCell(
            nn.Conv2d(in_channel, dims[0], kernel_size=4, stride=4),
            LayerNorm(normalized_shape=dims[0],
                      eps=1e-6,
                      data_format="channels_first"))
        self.downsample_layers.append(stem)

        # 构建stage2-stage4的前三个downsample
        for i in range(3):
            downsample_layer = nn.SequentialCell(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
        # 存储每一个stage所构建的一系列block
        self.stages = nn.CellList()
        linspace = P.LinSpace()
        start = Tensor(0, ms.float32)
        dp_rates = [x.item((0, )) for x in linspace(start, drop_path_rate, sum(depths))]
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

        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(init.initializer(init.TruncatedNormal(),
                                                      cell.weight.shape,
                                                      cell.weight.dtype))
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(cell.weight.data * head_init_scale)
                cell.bias.set_data(cell.bias.data * head_init_scale)

        self.reduce_mean = P.ReduceMean()

    def construct_feature(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(self.reduce_mean(x, [-2, -1]))      # global average pooling, (N, C, H, W) -> (N, C)

    def construct(self, x):
        x = self.construct_feature(x)
        x = self.head(x)
        return x


@ClassFactory.register(ModuleType.BACKBONE)
class ConvNeXtTiny(ConvNeXt):
    """
    This class of convnext_tiny.

    Examples:
        >>> from mindvision.classification.models.backbones import ConvNeXt
        >>> import numpy as np
        >>> net = ConvNeXtTiny(ConvNeXt)
        >>> x = np.random.randn(32*3*224*224).reshape(32, 3, 224, 224)
        >>> output = net(x)
        >>> print(output.shape)
    """
    def __init__(self, **kwargs):
        super(ConvNeXtTiny, self).__init__(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],  **kwargs)


@ClassFactory.register(ModuleType.BACKBONE)
class ConvNeXtSmall(ConvNeXt):
    """
    This class of convnext_small.

    Examples:
        >>> from mindvision.classification.models.backbones import ConvNeXt
        >>> import numpy as np
        >>> net = ConvNeXtSmall(ConvNeXt)
        >>> x = np.random.randn(32*3*224*224).reshape(32, 3, 224, 224)
        >>> output = net(x)
        >>> print(output.shape)
    """
    def __init__(self, **kwargs):
        super(ConvNeXtSmall, self).__init__(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)


@ClassFactory.register(ModuleType.BACKBONE)
class ConvNeXtBase(ConvNeXt):
    """
    This class of convnext_base.

    Examples:
        >>> from mindvision.classification.models.backbones import ConvNeXt
        >>> import numpy as np
        >>> net = ConvNeXTBase(ConvNeXt)
        >>> x = np.random.randn(32*3*224*224).reshape(32, 3, 224, 224)
        >>> output = net(x)
        >>> print(output.shape)
    """
    def __init__(self, **kwargs):
        super(ConvNeXtBase, self).__init__(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)


@ClassFactory.register(ModuleType.BACKBONE)
class ConvNeXtLarge(ConvNeXt):
    """
    This class of convnext_large.

    Examples:
        >>> from mindvision.classification.models.backbones import ConvNeXt
        >>> import numpy as np
        >>> net = ConvNeXtLarge(ConvNeXt)
        >>> x = np.random.randn(32*3*224*224).reshape(32, 3, 224, 224)
        >>> output = net(x)
        >>> print(output.shape)
    """
    def __init__(self, **kwargs):
        super(ConvNeXtLarge, self).__init__(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)


@ClassFactory.register(ModuleType.BACKBONE)
class ConvNeXtxLarge(ConvNeXt):
    """
    This class of convnext_xlarge.

    Examples:
        >>> from mindvision.classification.models.backbones import ConvNeXt
        >>> import numpy as np
        >>> net = ConvNeXtxLarge(ConvNeXt)
        >>> x = np.random.randn(32*3*224*224).reshape(32, 3, 224, 224)
        >>> output = net(x)
        >>> print(output.shape)
    """

    def __init__(self, **kwargs):
        super(ConvNeXtxLarge, self).__init__(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
