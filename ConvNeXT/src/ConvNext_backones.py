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

form mindvision.engine.class_factory import ClassFactory, ModuleType

__all__ = [
    "LayerNorm",
    "Block",
    "ConvNeXt",
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "convnext_large",
    "convnext_xlarge"
]
from src.drop_path import DropPath, Identity
from src.model_utils.moxing_adapter import config


class LayerNorm(nn.Cell):
    """
    LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).

    Args:
        normalized_shape: Normalization along axis.
        eps(float): A value added to the denominator for numerical stability. Default: 1e-7.
        data_format(str):channels_last corresponds to inputs with shape (batch_size, height, width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width).

    Returns:
        normalized x

    Examples:
        >>> LayerNorm(normalized_shape=96, eps=1e-6, data_format="channels_last")
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
    ConvNext block definition. There are two equivalent implementations:
    (1) DwConv -> layernorm(channel_first)->1*1 Conv —>GELU -> 1*1 Conv,all in (N, C, H, W);
    (2) DwConv -> Permute to (NHWC), layernorm(channels_last) -> Dense -> GELU -> Dense,
    permute back to (NCHW). We use (2).

    Args:
        dim(int): Number of input channels.
        drop_path(float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value(float): Init value for Layer Scale. Default: 1e-6.

    Returns:
        Tensor, x tensor.

    Examples:
        >>>from mindvision.classification.models.backbons import Block
        >>>Block(dim=96, drop_path=0.0, layer_scale_init_value=1e-6)
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super(Block, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, pad_mode="pad", padding=3, group=dim)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channel_last")
        self.pwconv1 = nn.Dense(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Dense(4 * dim, dim)
        self.gamma = Parameter(layer_scale_init_value * np.ones((dim)),
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
        title={},
        author={},
        journal={},
        year={}
        }
    """

    def __init__(self, in_channel=3,
                 num_classes=1000,
                 depths=[3, 3, 9, 3],
                 dims=[96, 192, 384, 768],
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 head_init_scale=1):
        super(ConvNeXt, self).__init__()
        print(type(dims[-1]))
        print(type(dims))
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
        ls = linspace(start, drop_path_rate, sum(depths))
        print("====================")
        print(ls)
        print("=====================")
        print(type(ls))
        dp_rates = [x.item((0, )) for x in linspace(start, drop_path_rate, sum(depths))]   # 等差数列
        print("=================================")
        print(dp_rates)
        print("==================================")
        print(type(dp_rates))
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

        if config.initialize_mode == "Trunc":
            # default_recurisive_init(self)
            self._init_weights()
        self.head.weight.data * head_init_scale
        self.head.bias.data * head_init_scale
        # self.head.weight.set_data(init.initializer(init.Constant(layer_scale_init_value),
        #                                            self.head.weight.shape,
        #                                            self.head.weight.dtype))
        # self.head.bias.set_data(init.initializer(init.Constant(layer_scale_init_value),
        #                                          self.head.bias.shape,
        #                                          self.head.bias.dtype))

    def _init_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, (nn.Conv2d, nn.Dense)):
                cell.weight.set_data(init.initializer(init.TruncatedNormal(),
                                                      cell.weight.shape,
                                                      cell.weight.dtype))
                # cell.bias.set_data(init.initializer('zeros',
                #                                     cell.bias.shape,
                #                                     cell.bias.dtype))

    def construct_feature(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))      # global average pooling, (N, C, H, W) -> (N, C)

    def construct(self, x):
        x = self.construct_feature(x)
        x = self.head(x)
        return x

@ClassFactory.register(ModuleType.BACKBONE)
class convnext_tiny(ConvNeXt):
    """
    This class of convnext_tiny.

    Examples:
        >>> from mindvision.classification.models.backbones import ConvNext
        >>> import numpy as np
        >>> net = convnext_tiny(ConvNext)
        >>> x = np.random.randn(32*3*224*224).reshape(32, 3, 224, 224)
        >>> output = net(x)
        >>> print(output.shape)
    """
    def __init__(self, **kwargs):
        super(convnext_tiny, self).__init__(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],  **kwargs)


@ClassFactory.register(ModuleType.BACKBONE)
class convnext_small(ConvNeXt):
    """
    This class of convnext_small.

    Examples:
        >>> from mindvision.classification.models.backbones import ConvNext
        >>> import numpy as np
        >>> net = convnext_small(ConvNext)
        >>> x = np.random.randn(32*3*224*224).reshape(32, 3, 224, 224)
        >>> output = net(x)
        >>> print(output.shape)
    """
    def __init__(self, **kwargs):
        super(convnext_small, self).__init__(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)


@ClassFactory.register(ModuleType.BACKBONE)
class convnext_base(ConvNeXt):
    """
    This class of convnext_base.

    Examples:
        >>> from mindvision.classification.models.backbones import ConvNext
        >>> import numpy as np
        >>> net = convnext_base(ConvNext)
        >>> x = np.random.randn(32*3*224*224).reshape(32, 3, 224, 224)
        >>> output = net(x)
        >>> print(output.shape)
    """
    def __init__(self, **kwargs):
        super(convnext_base, self).__init__(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)


@ClassFactory.register(ModuleType.BACKBONE)
class convnext_large(ConvNeXt):
    """
    This class of convnext_large.

    Examples:
        >>> from mindvision.classification.models.backbones import ConvNext
        >>> import numpy as np
        >>> net = convnext_large(ConvNext)
        >>> x = np.random.randn(32*3*224*224).reshape(32, 3, 224, 224)
        >>> output = net(x)
        >>> print(output.shape)
    """
    def __init__(self, **kwargs):
        super(convnext_large, self).__init__(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)


@ClassFactory.register(ModuleType.BACKBONE)
class convnext_xlarge(ConvNeXt):
    """
    This class of convnext_xlarge.

    Examples:
        >>> from mindvision.classification.models.backbones import ConvNext
        >>> import numpy as np
        >>> net = convnext_xlarge(ConvNext)
        >>> x = np.random.randn(32*3*224*224).reshape(32, 3, 224, 224)
        >>> output = net(x)
        >>> print(output.shape)
    """

    def __init__(self, **kwargs):
        super(convnext_xlarge, self).__init__(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)