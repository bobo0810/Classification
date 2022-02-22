import torchvision
from torchvision import transforms
from PIL import Image
import torch
import numpy as np

# ImageNet均值、方差
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


class PreProcess:
    """
    图像预处理入口
    """

    def __init__(self):
        pass

    @staticmethod
    def transforms(mode, img, img_shape=(224, 224)):
        """
        数据增广

        mode: 增广类型
        img: cv2读取的原图
        img_shape：训练图像尺寸
        """
        assert mode in ["train", "val", "test"]
        img = Image.fromarray(img)
        # =========================训练集========================================

        if mode == "train":
            img_transforms = transforms.Compose(
                [
                    transforms.Resize(img_shape),
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
        # =========================验证集/测试集==================================
        else:
            img_transforms = transforms.Compose(
                [
                    transforms.Resize(img_shape),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )

        return img_transforms(img)  # 增广

    @staticmethod
    def convert_vis(bchw_imgs, img_shape=(224, 224), vis_nums=8):
        """
        转化格式，方便tensorboard可视化
        1.反归一化 2.恢复通道顺序

        vis_nums: 最多显示的图像数
        """
        t_mean = torch.FloatTensor(mean).view(3, 1, 1).expand(3, 224, 224)
        t_std = torch.FloatTensor(std).view(3, 1, 1).expand(3, 224, 224)

        imgs = bchw_imgs[:vis_nums].clone()
        imgs = imgs * t_std + t_mean  # 反归一化
        imgs = imgs[:, [2, 1, 0], :, :]  # RGB->BGR
        imgs = torchvision.utils.make_grid(imgs)  # 拼成一张网格图 CHW
        imgs = imgs[:, 0::2, 0::2]  # 分辨率下降2倍
        return imgs
