import argparse as arg
import ast
import os
import zipfile

from mindspore import context, Model
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.common import set_seed
import moxing as mox

from src.load_dataset import create_dataset_imagenet

set_seed(1)
environment = 'train'
if environment == 'debug':
    workroot = '/home/ma-user/work'
else:
    workroot = '/home/work/user-job-dir'
print('current work mode:' + environment + ', workroot:' + workroot)

parser = arg.ArgumentParser(description="MindSpore ConvNext example")
parser.add_argument('--data_url',
                    help='path to training/inference dataset folder',
                    default=workroot + '/data/')
parser.add_argument('--train_url',
                    help='model folder to save/load',
                    default=workroot + '/model/')
parser.add_argument(
    '--device_target',
    type=str,
    default="Ascend",
    choices=['Ascend', 'CPU'],
    help='device where the code will be implemented (default: CPU),若要在启智平台上使用NPU，需要在启智平台训练界面上加上运行参数device_target=Ascend')
parser.add_argument('--epoch_size',
                    type=int,
                    default=5,
                    help='Training epochs.')


def zip_file(zip_src, dst_dir):
    r = zipfile.is_zipfile(zip_src)
    if r:
        fz = zipfile.ZipFile(zip_src, 'r')
        for file in fz.namelist():
            fz.extract(file, dst_dir)
        else:
            print('This is not zip')


if __name__ == "__main__":
    args = parser.parse_args()
    print('args:')
    print(args)
    data_dir = workroot + '/data'   # 数据集存放路径
    train_dir = workroot + '/model'  # 训练模型存放路径
    # 初始化数据存放目录
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    # 初始化模型存放目录
    obs_train_url = args.train_url
    train_dir = workroot + '/model/'
    if not os.path.exists(train_dir):
            os.mkdir(train_dir)
    # 将数据集从obs拷贝到镜像中
    if environment == 'train':
        obs_data_url =args.data_url
        try:
            mox.file.copy_parallel(obs_data_url, data_dir)
            print("Successfully Download {} to {}".format(obs_data_url,
                                                          data_dir))
        except Exception as e:
            print('moxing download {} to {} failed:'.format(obs_data_url,
                                                         data_dir) + str(e))
    path = data_dir + 'imagenet.zip'
    if zipfile.is_zipfile(path):
        print(123)
        print("====================================================")
        f = zipfile.ZipFile(path)
        files = f.namelist()
        print(files)
        print("====================================================")
    else:
        print(321)
    # 初始化存放解压缩文件目录
    # data_unzip_dir = workroot + '/unzip_data'
    # if not os.path.exists(data_unzip_dir):
    #     os.mkdir(data_unzip_dir)
    # zip_file(data_dir, data_unzip_dir)
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=args.device_target)

    # ds_train = create_dataset_imagenet(dataset_path=)
    # ds_train = create_dataset_imagenet(dataset_path=data_dir, batch_size=)


    # if not
    #
    # parser = arg.ArgumentParser(description="MindSpore ConvNext example")
    # parser.add_argument('--device_target', default='Ascend',
    #                     help='device where the code will be implemented')
    # parser.add_argument('--data_url', help='path to training/inference dataset folder',
    #                     default=workroot + '/data/')
    # parser.add_argument('--train_url', help='model folder to save/load',
    #                     default=workroot + '/model/')
    # parser.add_argument('--run_distribute', type=ast.literal_eval, required=False, default=None,
    #                     help='If run distributed')
    # args = parser.parse_args()
    # context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    #
    # if args.run_distribute:
    #     device_num = int(os.getenv('RANK_SIZE'))
    #     device_id = int(os.getenv('DEVICE_ID'))
    #     context.set_context(device_id=device_id)
    #     context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
    #                                       gradients_mean=True)
    #     init()
