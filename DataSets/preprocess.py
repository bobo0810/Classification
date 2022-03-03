import torchvision
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
from timm.data.transforms_factory import create_transform as timm_transform


class PreProcess:
    """
    图像预处理入口
    """

    def __init__(self):
        pass

    @staticmethod
    def transforms(mode, img, img_size=[224, 224]):
        """
        数据增广

        mode: 增广类型
        img: cv2读取的原图
        img_size：训练图像尺寸
        """
        assert mode in ["train", "val", "test"]
        img = Image.fromarray(img)
        # =========================训练集========================================
        if mode == "train":
            img_transforms = timm_transform(img_size, is_training=True)
        # =========================验证集/测试集==================================
        else:
            # resize256 -> centercrop224 -> ToTensor -> Normalize
            img_transforms = timm_transform(img_size)
        # print(img_transforms)
        return img_transforms(img)  # 增广

    @staticmethod
    def convert(imgs, names, per_nums=4):
        """
        转化格式，方便tensorboard可视化

        imgs(tensor): [B,C,H,W]
        names(list):[B]
        per_nums: batch内每类最多显示的图像数.默认为4
        """
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        t_mean = torch.FloatTensor(mean).view(3, 1, 1).expand(3, 224, 224)
        t_std = torch.FloatTensor(std).view(3, 1, 1).expand(3, 224, 224)

        # 按类别划分
        index_list = []
        for name in set(names):
            index_list.append(
                [i for i, name_i in enumerate(names) if name_i == name][:per_nums]
            )
        imgs_list = [imgs[index].clone() for index in index_list]

        # 反归一化 + RGB->BGR
        imgs_list = [(line * t_std + t_mean)[:, [2, 1, 0], :, :] for line in imgs_list]
        # 拼成网格图CHW
        imgs_list = [torchvision.utils.make_grid(line) for line in imgs_list]
        return imgs_list
