import os
import torch.utils.data as data
import torch
import cv2
from collections import Counter
from Utils.tools import analysis_dataset
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler
from pytorch_metric_learning import samplers
from .preprocess import *  # 导入预处理策略

cur_path = os.path.abspath(os.path.dirname(__file__))


class create_datasets(data.Dataset):
    """初始化数据集"""

    def __init__(self, txt, mode, size, process, use_augment=False):
        """
        txt:  数据集路径           eg:/home/xxx/dataset.txt
        mode: 加载的数据集类型      train:训练集  val:验证集  test:测试集
        size: 图像尺寸             eg: [224,224]
        process: 图像预处理的名称         eg:"ImageNet"
        use_augment: 是否开启图像增广
        """
        assert mode in ["train", "val", "test"]
        self.size = size
        self.mode = mode
        self.use_augment = use_augment

        # 解析数据集
        dataset = analysis_dataset(txt)
        self.imgs_list = dataset[mode]["imgs"]
        self.label_list = dataset[mode]["labels"]
        self.labels = dataset["labels"]

        # 预处理策略
        self.process = eval(process)

    def __getitem__(self, index):
        img_path = self.imgs_list[index]  # 图片路径
        category = self.label_list[index]  # 类别名称
        label = int(self.labels.index(category))  # 类别标签

        # 图像预处理
        assert os.path.exists(img_path), "图像不存在"
        cv2_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = self.process(cv2_img, self.size, self.use_augment)
        return img, label

    def __len__(self):
        return len(self.imgs_list)

    def get_labels(self):
        """
        构造 类别均衡的数据加载器，用于训练
        """
        return self.label_list

    def get_info(self):
        """
        获取类别信息
        """
        info = "The nums of %s: %d ." % (self.mode, len(self.imgs_list))
        info += "The nums of each class: %s." % dict(Counter(self.label_list))
        return info


def create_dataloader(batch_size, dataset, sampler_name="normal"):
    """
    初始化 数据加载器

    batch_size: 批次
    dataset: 数据集
    sampler_name: 均衡采样策略(仅训练集生效)
    """
    if dataset.mode == "train":  # 训练集
        assert sampler_name in ["normal", "dataset_balance", "batch_balance"]
        sampler = None  # 常规采样
        if sampler_name == "dataset_balance":  # 数据集均衡采样
            sampler = ImbalancedDatasetSampler(dataset)
        elif sampler_name == "batch_balance":  # batch均衡采样
            sampler = samplers.MPerClassSampler(
                labels=dataset.get_labels(),
                length_before_new_iter=len(dataset),
                m=4,  # batch内每个类别的数量
            )
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            shuffle=True if sampler is None else False,
            drop_last=True if sampler is None else False,
        )
    else:  # 验证集/测试集
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
    return dataloader
