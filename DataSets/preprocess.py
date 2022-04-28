"""
Author: your name
Date: 2022-04-25 17:36:28
LastEditTime: 2022-04-25 17:36:29
LastEditors: your name
Description: In User Settings Edit
FilePath: /Classification/DataSets/preprocess.py
"""
import torchvision
from torchvision import transforms
from PIL import Image
import torch
import cv2
import numpy as np
from Utils.tools import tensor2img
from timm.data.transforms_factory import create_transform as timm_transform
import os


class PreProcess:
    """
    图像预处理入口
    """

    def __init__(self):
        pass

    @staticmethod
    def transforms(img_path, is_training, img_size=[224, 224]):
        """
        数据增广
        img_path: 图像路径
        is_training: 是否开启图像增广
                True: random(缩放、裁剪、翻转、色彩...) -> ToTensor -> Normalize
                False：resize256 -> centercrop224 -> ToTensor -> Normalize
        img_size：训练图像尺寸
        """
        assert os.path.exists(img_path), "图像文件不存在"
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = Image.fromarray(img)
        if is_training:  # 训练集
            img_transforms = timm_transform(
                img_size,
                is_training=True,
                re_prob=0.5,
                re_mode="pixel",  # 随机擦除
                auto_augment=None,  # 自动增广  eg：rand-m9-mstd0.5    rand-m7-mstd0.5-inc1
            )
        else:  #  验证集/测试集
            img_transforms = timm_transform(img_size)
        return img_transforms(img)

    @staticmethod
    def convert(imgs, category, per_nums=4):
        """
        转化格式，方便tensorboard可视化

        imgs(tensor): 形状[B,C,H,W]
        category(list):形状[B]
        per_nums: batch内每类最多显示的图像数.默认为4
        """
        # 按类别划分
        index_list = []
        for name in set(category):
            index_list.append(
                [i for i, name_i in enumerate(category) if name_i == name][:per_nums]
            )
        imgs_list = [imgs[index].clone() for index in index_list]

        # 反归一化、RGB->BGR
        imgs_list = [tensor2img(imgs) for imgs in imgs_list]
        # 拼成网格图CHW
        imgs_list = [torchvision.utils.make_grid(line) for line in imgs_list]
        return imgs_list
