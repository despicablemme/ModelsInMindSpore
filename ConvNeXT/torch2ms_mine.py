# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""change torch pth to MindSpore ckpt example."""
import torch

from mindspore import Parameter
from mindspore import log as logger
from mindspore import save_checkpoint
from mindspore.common.initializer import initializer
from mindspore.common.tensor import Tensor


def torch_to_ms(ms_model, torch_model):
    """
    Updates mobilenetv2 model MindSpore param's data from torch param's data.
    Args:
        model: MindSpore model
        torch_model: torch model
    """

    print("Start loading.")
    # load torch parameter and MindSpore parameter
    # torch_param_dict = torch_model.state_dict()
    torch_param_dict = torch_model['model']
    ms_param_dict = ms_model.parameters_dict()

    for torch_key in torch_param_dict.keys():  # 每个dict
        key_split = torch_key.split('.')
        # ms_key = ''
        if key_split[0] == 'head':
            ms_key = key_split[0] + '.dense.' + key_split[1]

        elif key_split[0] == 'norm':
            ms_key = 'neck.layer_norm.' + trans_layernorm_param_name(key_split[1])

        elif key_split[0] == 'downsample_layers':
            if key_split[1] == '0':
                ms_key = 'backbone.start_cell.'
                if key_split[2] == '0':
                    ms_key += key_split[2] + '.' + key_split[3]
                else:
                    ms_key += key_split[2] + '.' + trans_layernorm_param_name(key_split[3])
            elif key_split[1] in ['1', '2', '3']:
                ms_num = str(int(key_split[1]) * 2 - 2)
                ms_key = 'backbone.down_sample_blocks.' + ms_num
                if key_split[1] == '0':
                    ms_key += '.layer_norm.' + trans_layernorm_param_name(key_split[3])
                else:
                    ms_key += '.conv.' + key_split[3]

        elif key_split[0] == 'stages':
            if key_split[1] == '0':
                ms_key = 'backbone.block1.' + key_split[2]
                if key_split[3] == 'norm':
                    key_split[3] = '.layer_norm.'
                ms_key += key_split[3]
                if len(key_split) > 5 & key_split[3] == '.layer_norm.':
                    ms_key += trans_layernorm_param_name(key_split[4])
                else:
                    ms_key += key_split[4]

            elif key_split[1] in ['1', '2', '3']:
                ms_num = str(int(key_split[1]) * 2 - 1)
                ms_key = 'backbone.down_sample_blocks.' + ms_num + '.' + key_split[2]
                if key_split[3] == 'norm':
                    key_split[3] = '.layer_norm.'
                ms_key += key_split[3]
                if len(key_split) > 5 & key_split[3] == '.layer_norm.':
                    ms_key += trans_layernorm_param_name(key_split[4])
                else:
                    ms_key += key_split[4]

        update_torch_to_ms(torch_param_dict, ms_param_dict, torch_key, ms_key)

    save_checkpoint(ms_param_dict, "convnext.ckpt")
    print("Finish load.")


def update_bn(torch_param_dict, ms_param_dict, ms_key, ms_key_tmp):
    """Updates MindSpore batch norm param's data from torch batch norm param's data."""

    str_join = '.'
    if ms_key_tmp[-1] == "moving_mean":
        ms_key_tmp[-1] = "running_mean"
        torch_key = str_join.join(ms_key_tmp)
        update_torch_to_ms(torch_param_dict, ms_param_dict, torch_key, ms_key)
    elif ms_key_tmp[-1] == "moving_variance":
        ms_key_tmp[-1] = "running_var"
        torch_key = str_join.join(ms_key_tmp)
        update_torch_to_ms(torch_param_dict, ms_param_dict, torch_key, ms_key)
    elif ms_key_tmp[-1] == "gamma":
        ms_key_tmp[-1] = "weight"
        torch_key = str_join.join(ms_key_tmp)
        update_torch_to_ms(torch_param_dict, ms_param_dict, torch_key, ms_key)
    elif ms_key_tmp[-1] == "beta":
        ms_key_tmp[-1] = "bias"
        torch_key = str_join.join(ms_key_tmp)
        update_torch_to_ms(torch_param_dict, ms_param_dict, torch_key, ms_key)


def update_torch_to_ms(torch_param_dict, ms_param_dict, torch_key, ms_key):
    """Updates MindSpore param's data from torch param's data."""

    value = torch_param_dict[torch_key].cpu().numpy()
    value = Parameter(Tensor(value), name=ms_key)
    _update_param(ms_param_dict[ms_key], value)


def _update_param(param, new_param):
    """Updates param's data from new_param's data."""

    def logger_msg():
        return f"Failed to combine the net and the parameters for param {param.name}."

    if isinstance(param.data, Tensor) and isinstance(new_param.data, Tensor):
        if param.data.dtype != new_param.data.dtype:
            logger.error(logger_msg())
            msg = ("Net parameters {} type({}) different from parameter_dict's({})"
                   .format(param.name, param.data.dtype, new_param.data.dtype))
            raise RuntimeError(msg)

        if param.data.shape != new_param.data.shape:
            if not _special_process_par(param, new_param):
                logger.error(logger_msg())
                msg = ("Net parameters {} shape({}) different from parameter_dict's({})"
                       .format(param.name, param.data.shape, new_param.data.shape))
                raise RuntimeError(msg)
            return

        param.set_data(new_param.data)
        return

    if isinstance(param.data, Tensor) and not isinstance(new_param.data, Tensor):
        if param.data.shape != (1,) and param.data.shape != ():
            logger.error(logger_msg())
            msg = ("Net parameters {} shape({}) is not (1,), inconsistent with parameter_dict's(scalar)."
                   .format(param.name, param.data.shape))
            raise RuntimeError(msg)
        param.set_data(initializer(new_param.data, param.data.shape, param.data.dtype))

    elif isinstance(new_param.data, Tensor) and not isinstance(param.data, Tensor):
        logger.error(logger_msg())
        msg = ("Net parameters {} type({}) different from parameter_dict's({})"
               .format(param.name, type(param.data), type(new_param.data)))
        raise RuntimeError(msg)

    else:
        param.set_data(type(param.data)(new_param.data))


def _special_process_par(par, new_par):
    """
    Processes the special condition.

    Like (12,2048,1,1)->(12,2048), this case is caused by GE 4 dimensions tensor.
    """
    par_shape_len = len(par.data.shape)
    new_par_shape_len = len(new_par.data.shape)
    delta_len = new_par_shape_len - par_shape_len
    delta_i = 0
    for delta_i in range(delta_len):
        if new_par.data.shape[par_shape_len + delta_i] != 1:
            break
    if delta_i == delta_len - 1:
        new_val = new_par.data.asnumpy()
        new_val = new_val.reshape(par.data.shape)
        par.set_data(Tensor(new_val, par.data.dtype))
        return True
    return False


def trans_layernorm_param_name(torch_name):
    """

    """
    if torch_name == 'weight':
        return 'gamma'
    elif torch_name == 'bias':
        return 'beta'


if __name__ == '__main__':
    # file path
    ms_model_path = ''
    torch_model_path = 'convnext_base_22k_1k_224.pth'
    # load model
    torch_model = torch.load(torch_model_path)
    #
    torch_to_ms(torch_model)
