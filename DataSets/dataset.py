import os
import sys
import torch.utils.data as data
import numpy as np
import random
import yaml
from .preprocess import PreProcess
import time
import glob
import torch
from collections import Counter

cur_path = os.path.abspath(os.path.dirname(__file__))


class create_datasets(data.Dataset):
    """加载数据集"""

    def __init__(self, cfg, mode, is_training=False):
        """
        mode:划分数据集
        is_training: 是否开启图像增广,默认不增广
        """
        assert mode in ["train", "val", "test"]
        self.labels = cfg["labels"]
        self.txt = cfg["txt"]
        self.size = cfg["size"]
        self.mode = mode
        self.is_training = is_training
        if not mode == "test":
            self.ratio = cfg["ratio"]
        # 读取图像列表
        imgs_list = open(self.txt, "r").readlines()
        imgs_list = [line.strip() for line in imgs_list if line.strip() != ""]  # 过滤空格行
        # 划分
        random.seed(227)
        random.shuffle(imgs_list)
        if mode == "train":
            self.imgs_list = imgs_list[: int(self.ratio * len(imgs_list))]
        elif mode == "val":
            self.imgs_list = imgs_list[int(self.ratio * len(imgs_list)) :]
        else:
            self.imgs_list = imgs_list

        self.category_list = []
        for img_path in self.imgs_list:
            label_name = img_path.split("/")[-2]
            self.category_list.append(label_name)

        print("*" * 28)
        print("The nums of %sSet: %d" % (mode, len(self.imgs_list)))
        print("The nums of each class: ", dict(Counter(self.category_list)), "\n")

    def __getitem__(self, index):
        img_path = self.imgs_list[index]  # 图片路径
        category = self.category_list[index]  # 类别名称
        label = int(self.labels.index(category))  # 类别标签
        image = PreProcess().transforms(img_path, self.is_training, self.size)  # 增广
        return image, label

    def __len__(self):
        return len(self.imgs_list)

    def get_labels(self):
        """
        构造 类别均衡的数据加载器，用于训练
        """
        return self.category_list
