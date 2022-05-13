
import mindspore as ms
from mindspore import context, Tensor
import numpy as np
from blocks import ConvNeXt
from conv_next import *


context.set_context(device_target="GPU")
context.set_context(mode=1)   # "PYNATIVE_MODE=1"

net = convnext_base()

test_input = Tensor(np.ones((1, 3, 224, 224)), ms.float32)

test_out = net(test_input)

print(test_out.shape)
