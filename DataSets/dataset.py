import os
import sys
import torch.utils.data as data
from PIL import Image
import cv2
import numpy as np
import random
import yaml
from torchvision import transforms
import time
import glob
import torch
from collections import Counter

cur_path = os.path.abspath(os.path.dirname(__file__))
# ImageNet均值、方差
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


class TrainSet(data.Dataset):
    """加载训练集和验证集"""

    def __init__(self, cfg, mode, txt_path=cur_path + "/../Config/train.txt"):
        assert mode in ["train", "val"]

        self.prefix = cfg["prefix"]
        self.ratio = cfg["ratio"]
        self.category = cfg["category"]

        # 加载数据集
        imgs_list = open(txt_path, "r").readlines()
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
            raise NotImplementedError

        self.labels_list, self.labels_name_list = (
            [self.category[img_path.split("/")[-2]] for img_path in self.imgs_list],
            [img_path.split("/")[-2] for img_path in self.imgs_list],
        )

        print("%s data have  %d imgs" % (mode, len(self.imgs_list)))
        print(
            "The distribution of each category: ", Counter(self.labels_name_list)
        )  # 统计各类数量

        if mode == "train":
            # 训练集
            self.data_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    # 翻转
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    # 旋转
                    transforms.RandomChoice(
                        [
                            # 在 (-a, a) 之间随机选择
                            transforms.RandomRotation(30),
                            transforms.RandomRotation(60),
                            transforms.RandomRotation(90),
                        ]
                    ),
                    # 颜色
                    transforms.RandomChoice(
                        [
                            transforms.ColorJitter(brightness=0.5),
                            transforms.ColorJitter(contrast=0.5),
                            transforms.ColorJitter(saturation=0.5),
                            transforms.ColorJitter(hue=0.5),
                            transforms.ColorJitter(
                                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                            ),
                            transforms.ColorJitter(
                                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3
                            ),
                            transforms.ColorJitter(
                                brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
                            ),
                        ]
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                    transforms.RandomErasing(),  # 遮挡
                ]
            )

        else:
            # 验证集
            self.data_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )

    def __getitem__(self, index):

        img_path = self.imgs_list[index]
        label = self.labels_list[index]

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = Image.fromarray(image)
        image = self.data_transform(image)

        return image, int(label)

    def __len__(self):
        return len(self.imgs_list)

    def get_labels(self):
        """
        用于构造类别均衡的数据加载器
        https://github.com/ufoym/imbalanced-dataset-sampler
        """
        return self.labels_list


class TestSet(data.Dataset):
    """加载测试集"""

    def __init__(self, cfg, txt_path=cur_path + "/../Config/test.txt"):
        # 接收参数
        self.prefix = cfg["prefix"]
        self.category = cfg["category"]

        # 加载数据集
        self.imgs_list = open(txt_path, "r").readlines()
        self.imgs_list = [
            self.prefix + line.strip() for line in self.imgs_list
        ]  # 图像完整路径=prefix+txt路径
        self.labels_list, self.labels_name_list = (
            [self.category[img_path.split("/")[-2]] for img_path in self.imgs_list],
            [img_path.split("/")[-2] for img_path in self.imgs_list],
        )
        print("test data have  %d imgs" % (len(self.imgs_list)))
        print(
            "The distribution of each category: ", Counter(self.labels_name_list)
        )  # 统计各类数量

        # 测试集
        self.data_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __getitem__(self, index):
        img_path = self.imgs_list[index]
        label = self.labels_list[index]

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = Image.fromarray(image)
        image = self.data_transform(image)
        return image, int(label)

    def __len__(self):
        return len(self.imgs_list)


# def test():
#     """测试示例"""
#     file = open(cur_path + "/../Config/train.yaml", "r")
#     cfg = yaml.load(file, Loader=yaml.FullLoader)

#     ======normal采样======
#     data_loader = torch.utils.data.DataLoader(
#         dataset=Train(cfg=cfg["DataSet"], mode="train"),
#         batch_size=8,
#         shuffle=True,
#         num_workers=4,
#         pin_memory=True,
#         drop_last=True,  # 无法整除时丢弃最后一批样本（影响BN）
#     )

#     ======类别平衡采样======
#     from torchsampler import ImbalancedDatasetSampler

#     data_set = Train(cfg=cfg["DataSet"], mode="train")
#     data_loader = torch.utils.data.DataLoader(
#         data_set,
#         sampler=ImbalancedDatasetSampler(data_set),
#         batch_size=8,
#         num_workers=4,
#         pin_memory=True,
#         # shuffle=True, # 禁用打乱
#         # drop_last=True,
#     )

#     for i, (imgs, labels) in enumerate(data_loader):
#         print()
