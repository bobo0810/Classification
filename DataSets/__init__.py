from torch.utils.data import DataLoader
from .dataset import create_datasets
from torchsampler import ImbalancedDatasetSampler
from pytorch_metric_learning import samplers


def create_dataloader(cfg, mode):
    """
    数据集加载器入口
    """
    is_training = True if mode == "train" else False
    dataset = create_datasets(cfg, mode, is_training)

    # 数据集加载器
    if mode == "train":
        # 训练集
        assert cfg["sampler"] in ["normal", "balance", "batch_balance"]
        if cfg["sampler"] == "normal":  # 常规采样
            sampler = None
        elif cfg["sampler"] == "dataset_balance":  # 数据集均衡采样
            sampler = ImbalancedDatasetSampler(dataset)
        elif cfg["sampler"] == "batch_balance":  # batch均衡采样
            sampler = samplers.MPerClassSampler(
                labels=dataset.get_labels(),
                length_before_new_iter=len(dataset),
                m=4,  # batch内每个类别的数量
            )
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=cfg["batch"],
            num_workers=4,
            pin_memory=True,
            shuffle=True if sampler is None else False,
            drop_last=True if sampler is None else False,
        )
    else:
        # 验证集/测试集
        dataloader = DataLoader(
            dataset,
            batch_size=cfg["batch"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
    return dataloader
