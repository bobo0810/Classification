from torch.utils.data import DataLoader
from .dataset import TrainSet, TestSet
from torchsampler import ImbalancedDatasetSampler


def DataSets(cfg, mode):
    """
    数据集入口
    加载train/val/test
    """
    assert mode in ["train", "val", "test"]

    if mode == "train":
        # 常规采样
        if cfg["sampler"] == "normal":
            dataloader = DataLoader(
                dataset=TrainSet(cfg, mode),
                batch_size=cfg["batch"],
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                drop_last=True,
            )
        # 类别均衡采样
        elif cfg["sampler"] == "balance":
            dataset = TrainSet(cfg, mode)
            dataloader = DataLoader(
                dataset,
                sampler=ImbalancedDatasetSampler(dataset),
                batch_size=cfg["batch"],
                num_workers=4,
                pin_memory=True,
                # drop_last=True,# 禁用
                # shuffle=True,# 禁用
            )
        else:
            raise NotImplementedError
    elif mode == "val":
        dataloader = DataLoader(
            dataset=TrainSet(cfg, mode),
            batch_size=cfg["batch"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
    elif mode == "test":
        dataloader = DataLoader(
            dataset=TestSet(cfg),
            batch_size=cfg["batch"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
    else:
        raise NotImplementedError
    return dataloader
