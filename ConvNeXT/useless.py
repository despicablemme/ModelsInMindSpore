import numpy as np
import mindspore as ms
from mindspore.nn import Dense
from mindspore import Tensor, Parameter


dense = Dense(4, 4)

print(dense.weight.data)

dense.weight.set_data(Parameter(Tensor(np.ones((4, 4)), dtype=ms.float32)))

w = dense.weight.data
print()