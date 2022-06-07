import mindspore.ops as P
from mindspore import nn


class DropPath(nn.Cell):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, keep_prob=None, seed=0):
        super(DropPath, self).__init__()
        self.keep_prob = 1 - keep_prob
        seed = min(seed, 0)
        self.rand = P.UniformReal(seed=seed)
        self.shape = P.Shape()
        self.floor = P.Floor()

    def construct(self, x):
        if self.training:
            x_shape = self.shape(x)
            random_tensor = self.rand((x_shape[0], 1, 1))
            random_tensor = random_tensor + self.keep_prob
            random_tensor = self.floor(random_tensor)
            x = x / self.keep_prob
            x = x * random_tensor

        return x


name = 'backbone.down_sample_blocks.1.0.dwconv.weight'
name_split = name.split('.')
print(name_split)
print(int(name_split[2]))
if name_split[2] in ['0', '1']:
    print(True)
