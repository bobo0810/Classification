import os
import sys
import torch.utils.data as data
import cv2
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

    def __init__(self, cfg, mode):
        assert mode in ["train", "val", "test"]
        self.prefix = cfg["prefix"]
        self.labels = cfg["labels"]
        self.txt = cfg["txt"]
        self.size = cfg["size"]
        self.mode = mode
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
        img_path = os.path.join(self.prefix, self.imgs_list[index])
        category = self.category_list[index]  # 类别名称
        label = int(self.labels.index(category))  # 类别标签

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = PreProcess().transforms(self.mode, image, self.size)  # 增广
        return image, label

    def __len__(self):
        return len(self.imgs_list)

    def get_labels(self):
        """
        训练集：用于构造类别均衡的数据加载器
        https://github.com/ufoym/imbalanced-dataset-sampler
        """
        return self.category_list
