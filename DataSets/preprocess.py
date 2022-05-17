from PIL import Image
import torch
import cv2
import os
import numpy as np
from timm.data.transforms_factory import create_transform as timm_transform

class PreProcess:
    """
    图像预处理入口
    """

    def __init__(self):
        pass

    @staticmethod
    def transforms(img_path, use_augment, img_size=[224, 224]):
        """
        数据增广
        img_path: 图像路径
        use_augment: 是否开启图像增广
                True: random(缩放、裁剪、翻转、色彩...) -> ToTensor -> Normalize
                False：resize256 -> centercrop224 -> ToTensor -> Normalize
        img_size：训练图像尺寸
        """
        assert os.path.exists(img_path), "图像文件不存在"
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = Image.fromarray(img)
        if use_augment:  # 训练集
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