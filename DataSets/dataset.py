import os
import sys
import torch.utils.data as data
import numpy as np
import random
import yaml
import time
import glob
import torch
from collections import Counter
from .preprocess import PreProcess
from Utils.tools import analysis_dataset
cur_path = os.path.abspath(os.path.dirname(__file__))


class create_datasets(data.Dataset):
    """加载数据集"""

    def __init__(self, cfg, mode, use_augment=False):
        """
        mode: 数据集类型  eg:train/val/test
        use_augment: 是否开启图像增广
        """
        assert mode in ["train", "val", "test"]
        self.size = cfg["size"]
        self.use_augment = use_augment

        # 解析数据集
        dataset = analysis_dataset(cfg["txt"])
        self.imgs_list = dataset[mode]["imgs"]
        self.label_list = dataset[mode]["labels"]
        self.labels = dataset["labels"]

        print("*" * 28)
        print("The nums of %sSet: %d" % (mode, len(self.imgs_list)))
        print("The nums of each class: ", dict(Counter(self.label_list)), "\n")

    def __getitem__(self, index):
        img_path = self.imgs_list[index]  # 图片路径
        category = self.label_list[index]  # 类别名称
        label = int(self.labels.index(category))  # 类别标签
        image = PreProcess().transforms(img_path, self.use_augment, self.size)  # 图像预处理
        return image, label

    def __len__(self):
        return len(self.imgs_list)

    def get_labels(self):
        """
        构造 类别均衡的数据加载器，用于训练
        """
        return self.label_list
