from blocks import ConvNeXt
import mindspore as ms
from mindspore import context, Tensor
import numpy as np


context.set_context(device_target="GPU")
context.set_context(mode=1)   # "PYNATIVE_MODE=1"

backbone = ConvNeXt(drop_path_rate=0.5)

test_input = Tensor(np.ones((1, 3, 224, 224)), ms.float32)

test_out = backbone(test_input)

print(test_out.shape)

# for m in backbone.cells_and_names():
#     print(m)
#     if m[0]:
#         print(m[0])

