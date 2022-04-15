from mindspore import nn, _checkparam
import mindspore.ops as o
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.nn.layer.activation import get_activation
from mindspore.nn.layer.conv import _Conv


class ComplexDense(nn.Cell):
    """
    具体错误校验参考Dense
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='normal',
                 bias_init='zeros',
                 has_bias=True,
                 activation=None):
        super(ComplexDense, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.has_bias = has_bias
        self.activation = get_activation(activation) if isinstance(activation, str) else activation
        self.activation_flag = self.activation is not None

        self.reshape = o.Reshape()
        # self.shape_op = o.shape()
        self.matmul = o.MatMul(transpose_b=True)

        self.weight_real = Parameter(initializer(weight_init, [out_channels, in_channels]), name="weight_real")
        self.weight_image = Parameter(initializer(weight_init, [out_channels, in_channels]), name="weight_image")

        self.bias_real = None
        self.bias_image = None
        if self.has_bias:
            self.bias_real = Parameter(initializer(bias_init, [out_channels]), name="bias_real")
            self.bias_image = Parameter(initializer(bias_init, [out_channels]), name="bias_image")
        self.bias_add = o.BiasAdd()

    def construct(self, x_real, x_image):
        y_real = self.matmul(x_real, self.weight_real) - self.matmul(x_image, self.weight_image)
        y_image = self.matmul(x_real, self.weight_image) + self.matmul(x_image, self.weight_real)
        if self.has_bias:
            y_real = self.bias_add(y_real, self.bias_real)
            y_image = self.bias_add(y_image, self.bias_image)
        if self.activation_flag:
            y_real = self.activation(y_real)
            y_image = self.activation(y_image)
        return y_real, y_image


class ComplexConv2d(_Conv):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 pad_mode='same',
                 padding=0,
                 dilation=1,
                 group=1,
                 has_bias=False,
                 weight_init='normal',
                 bias_init='zeros',
                 data_format='NCHW'):
        kernel_size = _checkparam.twice(kernel_size)
        stride = _checkparam.twice(stride)
        self._dilation = dilation
        dilation = _checkparam.twice(dilation)
        super(ComplexConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            pad_mode,
            padding,
            dilation,
            group,
            has_bias,
            weight_init,
            bias_init,
            data_format)
        self.conv2d = o.Conv2D(out_channel=self.out_channels,
                               kernel_size=self.kernel_size,
                               mode=1,
                               pad_mode=self.pad_mode,
                               pad=self.padding,
                               stride=self.stride,
                               dilation=self.dilation,
                               group=self.group,
                               data_format=self.format)
        self.bias_add = o.BiasAdd(data_format=self.format)

        # todo
        shape = [out_channels, *kernel_size, in_channels // group] if self.format == "NHWC" else \
            [out_channels, in_channels // group, *kernel_size]
        self.weight_image = Parameter(initializer(self.weight_init, shape), name='weight_image')
        if self.has_bias:
            self.bias_image = Parameter(initializer(self.bias_init, [out_channels]), name='bias_image')
        else:
            self.bias_image = None

    def construct(self, x_real, x_image):
        y_real = self.conv2d(x_real, self.weight) - self.conv2d(x_image, self.weight_image)
        y_image = self.conv2d(x_real, self.weight_image) + self.conv2d(x_image, self.weight)
        if self.has_bias:
            y_real = self.bias_add(y_real, self.bias)
            y_image = self.bias_add(y_image, self.bias_image)

        return y_real, y_image


class ComplexBatchNorm(nn.Cell):
    def __init__(self):
        super(ComplexBatchNorm, self).__init__()

    def construct(self, x_real, x_image):






























