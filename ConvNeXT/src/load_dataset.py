import os
from multiprocessing import cpu_count

from mindspore.communication.management import get_rank, get_group_size
import mindspore.common.dtype as mstype
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset as ds


def _get_rank_info():
    """get rank size and rank id"""
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = 1
        rank_id = 0
    return rank_size, rank_id


def create_dataset_imagenet(cfg, dataset_path, batch_size=32, repeat_num=1, training=True,
                            num_parallel_workers=16, shuffle=None, sampler=None, class_indexing=None):
    device_num, rank_id = _get_rank_info()
    if device_num == 1:
        num_parallel_workers = 96
        if num_parallel_workers > cpu_count():
            num_parallel_workers = cpu_count()
    else:
        ds.config.set_numa_enable(True)
    data_set = ds.ImageFolderDataset(dataset_dir=dataset_path,
                                     num_parallel_workers=4,
                                     shuffle=shuffle,
                                     sampler=sampler, class_indexing=class_indexing,
                                     num_shards=device_num,
                                     shard_id=rank_id)
    assert cfg.image_height == cfg.image_width, "imageney_cfg.image_height mot equal imagenet_cfg.image_width"
    image_size = cfg.image_height

    transform_img = []
    if training:
        transform_img = [
            CV.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            CV.RandomHorizontalFlip(prob=0.5)
        ]
    else:
        transform_img = [
            CV.Decode(),
            CV.Resize((256, 256)),
            CV.CenterCrop(image_size)
        ]

    data_set = data_set.map(input_columns="image", num_parallel_workers=num_parallel_workers,
                            operations=transform_img)

    data_set = data_set.batch(batch_size, drop_remainder=True)

    if repeat_num > 1:
        data_set = data_set.repeat(repeat_num)

    return data_set
