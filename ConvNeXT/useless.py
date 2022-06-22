# import mindspore.ops as P
# from mindspore import nn
import torch

#
# # Image Example
# N, C, H, W = 20, 5, 10, 10
# inp = torch.randn(N, C, H, W)
# # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
# # as shown in the image below
# layer_norm = nn.LayerNorm((C,), -1, -1)
# dict = layer_norm.parameters_dict()
#
#
# t_layer_norm = torch.nn.LayerNorm([N])
# pa = t_layer_norm.state_dict()
#
# output = layer_norm(inp)
