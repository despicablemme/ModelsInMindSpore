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

from mindspore import Parameter
from mindspore import log as logger
from mindspore import save_checkpoint
from mindspore.common.initializer import initializer
from mindspore.common.tensor import Tensor


def torch_to_ms(model, torch_model):
    """
    Updates mobilenetv2 model MindSpore param's data from torch param's data.
    Args:
        model: MindSpore model
        torch_model: torch model
    """

    print("Start loading.")
    # load torch parameter and MindSpore parameter
    torch_param_dict = torch_model.state_dict()
    ms_param_dict = model.parameters_dict()

    for ms_key in ms_param_dict.keys():  # 每个dict
        ms_key_tmp = ms_key.split('.')
        if ms_key_tmp[0] == "head":    # 先转移检测头参数
            if ms_key_tmp[-1] == "weight":
                update_torch_to_ms(torch_param_dict, ms_param_dict, "classifier.1.weight", ms_key)
            elif ms_key_tmp[-1] == "bias":
                update_torch_to_ms(torch_param_dict, ms_param_dict, "classifier.1.bias", ms_key)
        else:    # 不是头参数
            del (ms_key_tmp[0])
            str_join = '.'
            if ms_key_tmp[1] in ['0', '18']:
                del (ms_key_tmp[2])  # pylint: disable=superfluous-parens
                if ms_key_tmp[3] == '0':
                    torch_key = str_join.join(ms_key_tmp)
                    update_torch_to_ms(torch_param_dict, ms_param_dict, torch_key, ms_key)
                else:
                    update_bn(torch_param_dict, ms_param_dict, ms_key, ms_key_tmp)
            elif ms_key_tmp[1] == "1":
                if ms_key_tmp[4] == "feature":
                    del (ms_key_tmp[4])  # pylint: disable=superfluous-parens
                    if ms_key_tmp[4] == "0":
                        torch_key = str_join.join(ms_key_tmp)
                        update_torch_to_ms(torch_param_dict, ms_param_dict, torch_key, ms_key)
                    else:
                        update_bn(torch_param_dict, ms_param_dict, ms_key, ms_key_tmp)
                else:
                    if ms_key_tmp[3] == "1":
                        torch_key = str_join.join(ms_key_tmp)
                        update_torch_to_ms(torch_param_dict, ms_param_dict, torch_key, ms_key)
                    else:
                        update_bn(torch_param_dict, ms_param_dict, ms_key, ms_key_tmp)
            elif 2 <= int(ms_key_tmp[1]) <= 17:
                if ms_key_tmp[4] == "feature":
                    del (ms_key_tmp[4])  # pylint: disable=superfluous-parens
                    if ms_key_tmp[4] == "0":
                        torch_key = str_join.join(ms_key_tmp)
                        update_torch_to_ms(torch_param_dict, ms_param_dict, torch_key, ms_key)
                    else:
                        update_bn(torch_param_dict, ms_param_dict, ms_key, ms_key_tmp)
                else:
                    if ms_key_tmp[3] == "2":
                        torch_key = str_join.join(ms_key_tmp)
                        update_torch_to_ms(torch_param_dict, ms_param_dict, torch_key, ms_key)
                    else:
                        update_bn(torch_param_dict, ms_param_dict, ms_key, ms_key_tmp)

    save_checkpoint(model, "mobilenetv2.ckpt")
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
