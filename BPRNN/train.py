"""
1 创建model
2 准备数据
3 训练
4 推理
"""
import os
import argparse as arg
from mindspore.nn.loss import BCEWithLogitsLoss
from mindspore.nn.optim import Adam
from mindspore.nn.dynamic_lr import cosine_decay_lr
from mindspore import Model, context
from mindspore.train.callback import LossMonitor, TimeMonitor, ModelCheckpoint, CheckpointConfig

# import moxing as mox

from config import Configs
from get_code_config import load_code_message
from get_code_mat import get_mats
from dataset import Dataset
from layers import BPRNN
from utils import my_lr, BinaryAcc


def train_net(args):
    # configs init
    cs = Configs()

    # load files and generate code structure
    current_path = os.path.dirname(os.path.realpath(__file__))  # BootfileDirectory, 启动文件所在的目录
    project_root = os.path.dirname(current_path)  # 工程的根目录，对应ModelArts训练控制台上设置的代码目录
    H_file = os.path.join(project_root, cs.H_file)
    G_file = os.path.join(project_root, cs.G_file)

    H, G, n, m, k = load_code_message(H_file, G_file)
    vc_mat, cv_mat, llr_mat, llr_mat_trans, num_edges = get_mats(H, m, n)
    # init network ,loss, optimizers and others
    network = BPRNN(n, m/n, num_edges, cs.batch_size, llr_mat, llr_mat_trans, vc_mat, cv_mat, cs.iterations)
    loss = BCEWithLogitsLoss(reduction='mean')
    print(network.trainable_params())
    steps = int(cs.all_data_num / cs.batch_size)
    # lr = my_lr(args.learning_rate, steps*cs.epochs, args.decay, steps*args.decay_epoch)
    lr = cosine_decay_lr(args.min_lr, args.max_lr, steps*cs.epochs, steps, steps*args.decay_epoch)
    optimizer = Adam(network.trainable_params(), learning_rate=lr)
    model = Model(network, loss, optimizer, metrics=None)
    # set callbacks
    time_cb = TimeMonitor()
    loss_cb = LossMonitor(steps)
    config_ck = CheckpointConfig(save_checkpoint_steps=steps, keep_checkpoint_max=100)
    ckpoint_cb = ModelCheckpoint(prefix='bp-rnn', directory=cs.result_path, config=config_ck)
    callbacks_list = [time_cb, loss_cb, ckpoint_cb]
    # load train data
    data_loader = Dataset(cs.train_data_num, cs.test_data_num, cs.snr_l, cs.snr_h, cs.snr_step, cs.iterations, G, n, m)
    dataset = data_loader.get_train_data(phase='all_zero')
    dataset = dataset.batch(batch_size=cs.batch_size, drop_remainder=True)
    # train
    model.train(cs.epochs, dataset, callbacks=callbacks_list, dataset_sink_mode=False)


if __name__ == '__main__':

    parser = arg.ArgumentParser(description='Mindspore SID Example')
    parser.add_argument('--device_target', type=str, default='CPU',
                        help='device where the code will be implemented')
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--decay', type=float, default=0.95)
    parser.add_argument('--decay_epoch', type=int, default=10)
    parser.add_argument('--min_lr', type=float, default=0.0003)
    parser.add_argument('--max_lr', type=float, default=0.001)
    args = parser.parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)

    train_net(args)
