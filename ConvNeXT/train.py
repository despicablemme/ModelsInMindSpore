from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id, get_device_num, get_rank_id, get_job_id
from src.model import convnext_tiny
from src.load_dataset import create_dataset_cifar10, create_dataset_imagenet
from src.generator_lr import get_lr_imagenet
from src.utils.logging import get_logger

from mindspore import context, Tensor
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.context import ParallelMode
from mindspore.common import set_seed
import mindspore.nn as nn
from mindspore.train.model import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.nn.metrics import Accuracy

import os
import time

set_seed(1)


def modelarts_pre_process():
    ''' modelarts pre process function. '''
    def unzip(zip_file, save_dir):
        import zipfile
        # path = os.path.join(config.data_path, config.modelarts_dataset_unzip_name + ".zip")
        if zipfile.is_zipfile(zip_file):
            print(123)
            print("====================================================")
            f = zipfile.ZipFile(zip_file)
            files = f.namelist()
            print(files)
            print("====================================================")
        else:
            print(321)

        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, config.modelarts_dataset_unzip_name)):
            zip_isexist = zipfile.is_zipfile(zip_file)
            if zip_isexist:
                fz = zipfile.ZipFile(zip_file, 'r')
                data_num = len(fz.namelist())
                print("Extract Start...")
                print("unzip file num: {}".format(data_num))
                data_print = int(data_num / 100) if data_num > 100 else 1
                i = 0
                for file in fz.namelist():
                    if i % data_print == 0:
                        print("unzip percent: {}%".format(int(i * 100 / data_num)), flush=True)
                    i += 1
                    fz.extract(file, save_dir)
                print("cost time: {}min:{}s.".format(int((time.time() - s_time) / 60),
                                                     int(int(time.time() - s_time) % 60)))
                print("Extract Done.")
            else:
                print("This is not zip.")
        else:
            print("Zip has been extracted.")

    if config.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(config.data_path, config.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(config.data_path)

        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most.
        if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
            print("Zip file path: ", zip_file_1)
            print("Unzip file save dir: ", save_dir_1)
            unzip(zip_file_1, save_dir_1)
            print("===Finish extract data synchronization===")
            try:
                os.mknod(sync_lock)
            except IOError:
                pass

        while True:
            if os.path.exists(sync_lock):
                break
            time.sleep(1)

        print("Device: {}, Finish sync unzip data from {} to {}.".format(get_device_id(), zip_file_1, save_dir_1))

    config.ckpt_save_dir = os.path.join(config.output_path, config.ckpt_save_dir)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_train():
    ''' run train '''
    cfg = config
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target)
    device_num = get_device_num()

    if cfg.device_target == "Ascend":
        device_id = get_device_id()
        context.set_context(device_id=device_id)
        if device_num > 1:
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            init()
    elif cfg.device_target == "GPU":
        context.set_context(enable_graph_kernel=True)
        device_id = 0
        if device_num > 1:
            init()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            device_id = get_rank()
    print("init finished")

    if config.dataset_name == "imagenet":
        ds_train = create_dataset_imagenet(cfg=config,
                                           dataset_path=config.train_data_path,
                                           batch_size=config.batch_size)
    else:
        raise ValueError("Unsupported dataset.")

    if ds_train.get_dataset_size() == 0:
        raise ValueError("Please check dataset size > 0 and batch_size <= dataset_size")

    net = convnext_tiny(pretrained=False,
                        in_22k=False)
    step_per_epoch = ds_train.get_dataset_size()

    if config.dataset_name == 'imagenet':
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        lr = Tensor(get_lr_imagenet(config.learning_rate, config.epoch_size, step_per_epoch))
        opt = nn.Adam(params=net.trainable_params(),
                      learning_rate=lr,   # 4e-3
                      beta1=0.9,
                      beta2=0.999,
                      eps=1e-7,
                      weight_decay=0.05,
                      loss_scale=1.0)

    metrics = {"Accuracy": Accuracy()}
    model = Model(net, loss_fn=loss, optimizer=opt, metrics=metrics)

    time_cb = TimeMonitor()
    loss_cb = LossMonitor(per_print_times=step_per_epoch )
    callbacks_list = [time_cb, loss_cb]
    config_ck = CheckpointConfig(save_checkpoint_steps=step_per_epoch * 10,
                                 keep_checkpoint_max=300)
    if get_rank_id() == 0:
        ckpoint_cb = ModelCheckpoint(prefix='Vgg16_train', directory=config.ckpt_save_dir, config=config_ck)
        callbacks_list.append(ckpoint_cb)

    print("train start!")
    model.train(epoch=300,
                train_dataset=ds_train,
                callbacks=callbacks_list,
                dataset_sink_mode=config.dataset_sink_mode)


if __name__ == '__main__':
    run_train()
