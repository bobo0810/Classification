import os
import torch.utils.data as data
import torch
from collections import Counter
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler
from pytorch_metric_learning import samplers
from .preprocess import preprocess

cur_path = os.path.abspath(os.path.dirname(__file__))


class create_datasets(data.Dataset):
    """初始化数据集"""

    def __init__(self, dataset, size, process, use_augment=False):
        """
        dataset: 数据集信息
        size: 图像尺寸             eg: [224,224]
        process: 图像预处理的名称    eg:"ImageNet"
        use_augment: 是否图像增广
        """
        self.size = size
        self.use_augment = use_augment
        self.process = process

        self.imgs_list = dataset["imgs"]
        if "labels" in dataset.keys():
            self.label_list = dataset["labels"]
            self.all_labels = dataset["all_labels"]
        else:
            self.label_list = []  # 样本对，无label

    def __getitem__(self, index):
        img_path = self.imgs_list[index]  # 图片路径
        img = preprocess(self.process, img_path, self.size, self.use_augment)

        if self.label_list == []:
            return img, img_path
        else:
            label = self.label_list[index]  # 类别名
            label = int(self.all_labels.index(label))  # 类别ID
            return img, label

    def __len__(self):
        return len(self.imgs_list)

    def get_labels(self):
        """
        构造 类别均衡的训练集
        """
        return self.label_list


def create_dataloader(batch_size, dataset, sampler_name=None):
    """
    初始化 数据加载器

    batch_size: 批次
    dataset: 数据集
    sampler_name: 均衡采样策略(仅训练集)
    """
    if sampler_name is None:  # 验证集/测试集
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
    else:  # 训练集
        assert sampler_name in ["normal", "dataset_balance", "batch_balance"]
        if sampler_name == "dataset_balance":  # 数据集均衡采样
            sampler = ImbalancedDatasetSampler(dataset)
        elif sampler_name == "batch_balance":  # batch均衡采样
            sampler = samplers.MPerClassSampler(
                labels=dataset.get_labels(),
                length_before_new_iter=len(dataset),
                m=4,  # batch内每个类别的数量
            )
        else:
            sampler = None
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            shuffle=True if sampler is None else False,
            drop_last=True if sampler is None else False,
        )
    return dataloader
