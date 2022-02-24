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


class ImgSet(data.Dataset):
    """加载数据集"""

    def __init__(self, cfg, mode):
        assert mode in ["train", "val", "test"]
        self.prefix = cfg["prefix"]
        self.category = cfg["category"]
        self.txt = cfg["txt"]
        self.mode = mode
        if not mode == "test":
            self.ratio = cfg["ratio"]

        # 读取图像列表
        imgs_list = open(self.txt, "r").readlines()
        imgs_list = [
            self.prefix + line.strip() for line in imgs_list
        ]  # 图像完整路径=prefix+txt路径

        # 划分
        random.shuffle(imgs_list)
        if mode == "train":
            self.imgs_list = imgs_list[: int(self.ratio * len(imgs_list))]
        elif mode == "val":
            self.imgs_list = imgs_list[int(self.ratio * len(imgs_list)) :]
        else:
            self.imgs_list = imgs_list

        self.labels_list, self.labels_name_list = (
            [self.category[img_path.split("/")[-2]] for img_path in self.imgs_list],
            [img_path.split("/")[-2] for img_path in self.imgs_list],
        )

        print("*" * 28)
        print("The nums of %sSet: %d" % (mode, len(self.imgs_list)))
        print("The nums of each class: ", dict(Counter(self.labels_name_list)), "\n")

    def __getitem__(self, index):

        img_path = self.imgs_list[index]
        label = self.labels_list[index]
        name = self.labels_name_list[index]

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = PreProcess().transforms(self.mode, image)  # 增广
        return image, int(label), name

    def __len__(self):
        return len(self.imgs_list)

    def get_labels(self):
        """
        训练集：用于构造类别均衡的数据加载器
        https://github.com/ufoym/imbalanced-dataset-sampler
        """
        return self.labels_list
