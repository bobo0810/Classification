from torch.utils.data import DataLoader
from .dataset import ImgSet
from torchsampler import ImbalancedDatasetSampler


def DataSets(cfg, mode):
    """
    数据集入口
    加载train/val/test
    """
    assert mode in ["train", "val", "test"]

    if mode == "train":  # 训练集
        # 常规采样
        if cfg["sampler"] == "normal":
            dataloader = DataLoader(
                dataset=ImgSet(cfg, mode),
                batch_size=cfg["batch"],
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                drop_last=True,
            )
        # 类别均衡采样
        elif cfg["sampler"] == "balance":
            dataset = ImgSet(cfg, mode)
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
    else:  # 验证集/测试集
        dataloader = DataLoader(
            dataset=ImgSet(cfg, mode),
            batch_size=cfg["batch"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
    return dataloader
