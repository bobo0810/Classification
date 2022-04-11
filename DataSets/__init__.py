from torch.utils.data import DataLoader
from .dataset import create_datasets
from torchsampler import ImbalancedDatasetSampler


def create_dataloader(cfg, mode):
    """
    数据集加载器入口
    加载train/val/test
    """
    is_training = True if mode == "train" else False
    dataset = create_datasets(cfg, mode, is_training)

    # 数据集加载器
    if mode == "train":  # 训练集
        # 常规采样
        if cfg["sampler"] == "normal":
            dataloader = DataLoader(
                dataset,
                batch_size=cfg["batch"],
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                drop_last=True,
            )
        # 类别均衡采样（每轮重新采样）
        elif cfg["sampler"] == "balance":
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
            dataset,
            batch_size=cfg["batch"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
    return dataloader
