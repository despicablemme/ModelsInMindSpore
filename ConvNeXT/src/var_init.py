import mindspore.nn as nn
from mindspore.common import initializer as init

import math
from functools import reduce


def _calculate_in_and_out(arr):
    """
    Calculate n_in and n_out.

    Args:
        arr (Array): Input array.

    Returns:
        Tuple, a tuple with two elements, the first element is `n_in` and the second element is `n_out`.
    """
    dim = len(arr.shape)
    if dim < 2:
        raise ValueError("If initialize data with xavier uniform, the dimension of data must greater than 1.")

    n_in = arr.shape[1]
    n_out = arr.shape[0]

    if dim > 2:
        counter = reduce(lambda x, y: x * y, arr.shape[2:])
        n_in *= counter
        n_out *= counter
    return n_in, n_out


def default_recurisive_init(custom_cell):
    """default_recurisive_init"""
    for _, cell in custom_cell.cells_and_names():
        if isinstance(cell, nn.Conv2d):
            cell.weight.set_data(init.initializer(init.TruncatedNormal(),
                                                  cell.weight.shape,
                                                  cell.weight.dtype))

            fan_in, _ = _calculate_in_and_out(cell.weight)
            # bound = 1 / math.sqrt(fan_in)
            # if cell.bias is not None:
            #     cell.bias.set_data(init.initializer(init.Uniform(bound),
            #                                         cell.bias.shape,
            #                                         cell.bias.dtype))
        elif isinstance(cell, nn.Dense):
            cell.weight.set_data(init.initializer(init.TruncatedNormal(),
                                                  cell.weight.shape,
                                                  cell.weight.dtype))
            fan_in, _ = _calculate_in_and_out(cell.weight)
            bound = 1 / math.sqrt(fan_in)
            if cell.bias is not None:
                cell.bias.set_data(init.initializer(init.Uniform(bound),
                                                    cell.bias.shape,
                                                    cell.bias.dtype))
        elif isinstance(cell, (nn.BatchNorm2d, nn.BatchNorm1d)):
            pass