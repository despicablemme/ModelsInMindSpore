

def get_group(network, args, lr_scheduler, step_per_epoch, assigner=None, weight_decay=1e-5, skip_list=[]):
    """

    """
    param_groups = {}

    for (name, param) in network.parameters_and_names():
        if len(param.shape) == 1 or name.endswith('.bias') or name in skip_list:
            group_name = 'no_decay'
            this_weight_decay = 0.
        else:
            group_name = 'decay'
            this_weight_decay = weight_decay

        layer_id = get_param_id(name)
        lr_scale = 1.
        if assigner is not None:
            lr_scale = assigner.get_lr_scale(layer_id)
        group_name = "layer_%d_%s" % (layer_id, group_name)
        if group_name not in param_groups:
            param_groups[group_name] = {'params': [],
                                        'weight_decay': this_weight_decay,
                                        'lr': lr_scheduler(lr=args.lr * lr_scale,
                                                           steps_per_epoch=step_per_epoch,
                                                           warmup_epochs=args.warmup_epochs,
                                                           max_epoch=args.epoch_size,
                                                           t_max=150,
                                                           eta_min=0)}

        param_groups[group_name]['params'].append(param)

    return list(param_groups.values())


def get_param_id(name):
    """

    """
    name_split = name.split('.')

    layer_id = 13
    if name_split[0] == 'backbone':
        if name_split[1] == 'start_cell':   # name=backbone.start_cell.0.weight
            layer_id = 0
        elif name_split[1] == 'block1':        # name=backbone.block1.0.dwconv.weight
            layer_id = 1
        elif name_split[1] == 'down_sample_blocks':   # name=backbone.down_sample_blocks.0.layer_norm.gamma
            if name_split[2] in ['0', '1']:               # name=backbone.down_sample_blocks.1.0.dwconv.weight
                layer_id = 2
            elif name_split[2] == '2':
                layer_id = 3
            elif name_split[2] == '3':
                layer_id = 3 + int(name_split[3]) // 3
            elif name_split[2] in ['4', '5']:
                layer_id = 12

    return layer_id


class ParamLRValueAssigner:
    """
    values: param decay values with length 14 for 14 levels
    """
    def __init__(self, values):
        self.values = values

    def get_lr_scale(self, layer_id):
        """
        get lr scale value for given layer_id
        """
        return self.values[layer_id]
