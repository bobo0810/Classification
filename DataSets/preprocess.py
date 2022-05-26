from PIL import Image
import torch
import cv2
import os
import numpy as np
import torchvision
from torchvision.transforms import transforms
from timm.data.transforms_factory import create_transform as timm_transform

# 当前支持
Process_list = [
    "ImageNet",  # ImageNet预处理
    "FaceCompare",  # 人脸比对预处理
]


class create_process:
    def __init__(self, name):
        """
        初始化图像预处理

        name: 预处理名称
        """
        assert name in Process_list, "NotImplementedError"
        self.name = name

    def __call__(self, cv2_img, img_size, use_augment=False):
        """
        执行预处理

        cv2_img: cv2读取的图像numpy
        img_size:训练图像尺寸
        use_augment:是否图像增广
        """
        if self.name == "ImageNet":
            img = self.ImageNet(cv2_img, img_size, use_augment)
        elif self.name == "FaceCompare":
            img = self.ImageNet(cv2_img, img_size, use_augment)
        return img

    @staticmethod
    def ImageNet(cv2_img, img_size, use_augment):
        """
        ImageNet预处理
        """
        img = Image.fromarray(cv2_img)
        if use_augment:
            # 增广：Random(缩放、裁剪、翻转、色彩...)
            img_transforms = timm_transform(
                img_size,
                is_training=True,
                re_prob=0.5,
                re_mode="pixel",  # 随机擦除
                auto_augment=None,  # 自动增广  eg：rand-m9-mstd0.5
            )
        else:
            # 不增广：ReSize256 -> CenterCrop224
            img_transforms = timm_transform(img_size)
        return img_transforms(img)

    @staticmethod
    def FaceCompare(cv2_img, img_size, use_augment):
        """
        人脸比对预处理
        注：人脸比对数据集 默认已基于关键点裁剪对齐。 故img_size暂不使用
        """
        img = Image.fromarray(cv2_img)

        # ImageNet均值方差
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if use_augment:
            # 增广
            img_transforms = torchvision.transforms.Compose(
                [
                    # TODO 噪声、模糊
                    torchvision.transforms.RandomHorizontalFlip(),  # 镜像翻转
                    torchvision.transforms.ColorJitter(
                        brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
                    ),  # 色彩抖动
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=mean, std=std),
                ]
            )

        else:
            # 不增广
            img_transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=mean, std=std),
                ]
            )
        return img_transforms(img)
