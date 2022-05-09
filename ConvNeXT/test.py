# from src.model import convnext_tiny, ConvNeXt
# import numpy as np
#
# x = np.random.randn(32*224*224*3).reshape(32, 224, 224, 3)
#
# # net = convnext_tiny(pretrained=False,
# #                     in_22k=False)
# # print("2222222222222222")
# net = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])
# print("1111111111111111111")
# net(x)
# for m in net.trainable_params():
#     print(m)
from src.model import ConvNeXt
import numpy as np
#
# x = np.random.randn(32*3*224*224).reshape(32, 3, 224, 224)
# net = ConvNeXt(in_channel=3, num_classes=10,
#                depths=[3, 3, 9, 3], dims=[96, 192, 284, 768],
#                drop_path_rate=0, layer_scale_init_value=1e-6,
#                head_init_scale=1)
# net(x)
# from src.model import LayerNorm
# LayerNorm(normalized_shape=96, eps=1e-6, data_format="channels_last")
#
# from mindspore.numpy import empty
# import numpy as np
#
# x = np.empty([2, 3])
# print(x)

import mindspore.nn.probability.distribution as msd

keep_brop = 0.5
a = msd.bernoulli(keep_brop)