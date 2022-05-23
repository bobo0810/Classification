from PIL import Image
import torch
import cv2
import os
import numpy as np
import torchvision
from torchvision.transforms import transforms
from timm.data.transforms_factory import create_transform as timm_transform
preprocess_list=[
    ImageNet_PreProcess,# ImageNet预处理
    Face_PreProcess,# 人脸比对预处理
]

def ImageNet_PreProcess(img_path, use_augment=False, img_size=[224, 224]):
    '''
    ImageNet预处理

    img_path: 图像路径
    use_augment: 是否图像增广
    img_size：训练图像尺寸
    '''
    assert os.path.exists(img_path), "图像文件不存在"
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = Image.fromarray(img)
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

def Face_PreProcess(img_path, use_augment=False, img_size=[112, 112]):
    '''
    人脸比对预处理
    注：人脸比对数据集 默认已基于关键点裁剪并对齐。

    img_path: 图像路径
    use_augment:是否图像增广
    img_size:训练图像尺寸
    '''
    assert os.path.exists(img_path), "图像文件不存在"
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = Image.fromarray(img)

    if use_augment: 
        # 增广 
        img_transforms = torchvision.transforms.Compose(
            [   
                # TODO 噪声、模糊
                torchvision.transforms.RandomHorizontalFlip(), # 镜像翻转
                torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), # 色彩抖动
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.5,), std=(0.5,)),
            ]
        )

    else: 
        # 不增广
        img_transforms = torchvision.transforms.Compose(
            [   
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.5,), std=(0.5,)),
            ]
        )
    return img_transforms(img)
