import mindspore as ms
import mindspore.nn.probability.distribution as msd
from mindspore import nn, Parameter, Tensor
from mindspore.ops import operations as P
import numpy as np


class ConvNeXt(nn.Cell):
    """
    Args:
        in_channels(int): Number of input image channels. Default: 3
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self,
                 in_channels=3,
                 depths=None,
                 dims=None,
                 im_size=224,
                 drop_path_rate=0.,
                 layer_scale=1e-6):
        super(ConvNeXt, self).__init__()
        if not depths:
            depths = [3, 3, 9, 3]
        if not dims:
            dims = [96, 192, 384, 768]
        im_size = im_size // 4
        self.start_cell = nn.SequentialCell([nn.Conv2d(in_channels, dims[0], 4, 4),
                                             nn.LayerNorm(normalized_shape=(dims[0], im_size, im_size),
                                                          begin_norm_axis=1,
                                                          begin_params_axis=1)])

        linspace = P.LinSpace()
        start = Tensor(0, ms.float32)
        dp_rates = [x.item((0,)) for x in linspace(start, drop_path_rate, sum(depths))]
        print(len(dp_rates))       # TODO
        print("dp_rates", dp_rates)

        self.block1 = nn.SequentialCell([ConvNeXtBlock(dim=dims[0],
                                                       out_size=im_size,
                                                       drop_path=dp_rates[j],
                                                       layer_scale=layer_scale)
                                         for j in range(depths[0])])
        del dp_rates[: depths[0]]
        print(len(dp_rates))

        down_sample_blocks_list = []
        for i in range(3):
            down_sample = DownSample(in_channels=dims[i], out_channels=dims[i+1], out_size=im_size)
            im_size = im_size // 2      # todo
            down_sample_blocks_list.append(down_sample)
            block = nn.SequentialCell([ConvNeXtBlock(dim=dims[i+1],
                                                     out_size=im_size,
                                                     drop_path=dp_rates[j],
                                                     layer_scale=layer_scale)
                                       for j in range(depths[i+1])])
            down_sample_blocks_list.append(block)
            del dp_rates[: depths[i+1]]
            print(len(dp_rates), i)
        self.down_sample_blocks = nn.SequentialCell(down_sample_blocks_list)

    def construct(self, x):
        x = self.start_cell(x)
        x = self.block1(x)
        x = self.down_sample_blocks(x)
        return x


class ConvNeXtBlock(nn.Cell):
    """
    ConvNext Block. There are two equivalent implementations:
    (1) DwConv -> layernorm(channel_first)->1*1 Conv ???>GELU -> 1*1 Conv,all in (N, C, H, W);
    (2) DwConv -> Permute to (NHWC), layernorm(channels_last) -> Dense -> GELU -> Dense,
    permute back to (NCHW). We use (2).

    Args:
        dim(int):Number of input channels.
        drop_path(float): Stochastic depth rate. Default:0.0
        layer_scale(float): Init value for Layer Scale. Default:1e-6

    Returns:
        tensor

    Examples:
        >>> ConvNeXtBlock(dim=96, out_size=56, drop_path=0.0, layer_scale=1e-6)
    """
    def __init__(self,
                 dim: int,
                 out_size: int,
                 drop_path: float = 0.0,
                 layer_scale: float = 1e-6):
        super(ConvNeXtBlock, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, pad_mode="pad", padding=3, group=dim)
        self.layer_norm = nn.LayerNorm(normalized_shape=(dim, out_size, out_size),
                                       begin_norm_axis=1,
                                       begin_params_axis=1,
                                       epsilon=1e-6)
        self.transpose = P.Transpose()
        self.pwconv1 = nn.Dense(dim, 4 * dim)
        self.acti = nn.GELU()
        self.pwconv2 = nn.Dense(4 * dim, dim)
        # todo
        if layer_scale > 0:
            self.gamma = Parameter(Tensor(layer_scale * np.ones((dim,)), dtype=ms.float32), requires_grad=True)
        else:
            self.gamma = Parameter(Tensor(np.ones((dim,)), dtype=ms.float32), requires_grad=False)
        self.drop_path = DropPathConvNeXt(drop_path)

    def construct(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.layer_norm(x)
        x = self.transpose(x, (0, 2, 3, 1))
        x = self.pwconv1(x)
        x = self.acti(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = self.transpose(x, (0, 3, 1, 2))
        x = shortcut + self.drop_path(x)
        return x


class DownSample(nn.Cell):
    """
    down sample block.

    Args:
        in_channels(int): Number of input channels.
        out_channels(int): Number of output channels.
        out_size(int): image size after processing.
        kernel_size(int): Convolution kernel size.
        stride(int): stride size.
        eps(float): A value added to the denominator for numerical stability. Default: 1e-7.

    Returns:
        Tensor

    Examples:
        >>> DownSample(in_channels=96, out_channels=96, out_size=112, kernel_size=2, stride=2, eps=1e-6)
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 out_size: int,
                 kernel_size: int = 2,
                 stride: int = 2,
                 eps: float = 1e-6,
                 ):
        super(DownSample, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=(in_channels, out_size, out_size),  # todo
                                       begin_norm_axis=1,
                                       begin_params_axis=1,
                                       epsilon=eps)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def construct(self, x):
        """DownSample forward construct"""
        x = self.layer_norm(x)
        x = self.conv(x)
        return x


class DropPathConvNeXt(nn.Cell):
    """
    DropPath function for ConvNeXt.

    Args:
        drop_prob(float): Drop rate, (0, 1). Default:0.0
        scale_by_keep(bool): Determine whether to scale. Default: True.

    Returns:
        Tensor
    """

    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super(DropPathConvNeXt, self).__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1.0 - self.drop_prob
        if self.keep_prob == 1.0:
            self.keep_prob = 0.9999
        self.scale_by_keep = scale_by_keep
        self.bernoulli = msd.Bernoulli(probs=self.keep_prob)
        self.div = P.Div()

    def construct(self, x):
        if self.drop_prob > 0.0 and self.training:
            random_tensor = self.bernoulli.sample((x.shape[0],) + (1,) * (x.ndim - 1))
            if self.keep_prob > 0.0 and self.scale_by_keep:
                random_tensor = self.div(random_tensor, self.keep_prob)
            x = x * random_tensor

        return x
