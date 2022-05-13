
from mindspore import nn
import mindspore.ops as p


class Residual(nn.Cell):
    """
        function: an initialized Cell used in residual
    """
    def __init__(self, function):
        super(self, Residual).__init__()
        self.function = function

    def construct(self, x):
        return x + self.function(x)


class Block(nn.Cell):
    def __init__(self, num_channels, depth, kernel_size):
        super(Block, self).__init__()
        self.block = nn.SequentialCell([
                         nn.SequentialCell([
                             Residual(nn.SequentialCell([nn.Conv2d(num_channels,
                                                                   num_channels,
                                                                   kernel_size,
                                                                   pad_mode='same'),
                                                         nn.GELU(),
                                                         nn.BatchNorm2d(num_channels)])
                                      ),
                             nn.Conv2d(num_channels, num_channels, 1),
                             nn.GELU(),
                             nn.BatchNorm2d(num_channels)]) for i in range(depth)])

    def construct(self, x):
        return self.block(x)


class ConvMixer(nn.Cell):
    def __init__(self, dim, depth, kernel_size=9, patch_size=7, num_classes=1000):
        self.start = nn.SequentialCell([nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
                                        nn.GELU(),
                                        nn.BatchNorm2d(dim)])
        self.blocks = Block(num_channels=dim, depth=depth, kernel_size=kernel_size)
        self.adaptive_pool = p.AdaptiveAvgPool2D((1, 1))
        self.flatten = nn.Flatten()
        self.head = nn.Dense(num_classes, num_classes)

    def construct(self, x):
        x = self.start(x)
        x = self.blocks(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.head(x)
        return x
