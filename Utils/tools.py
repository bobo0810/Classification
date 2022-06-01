import torch
import math
import random
import time
import os
import cv2
import sys
import torchvision
from bobotools.txt_tools import TXT_Tools
from bobotools.img_tools import Img_Tools
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np


cur_path = os.path.abspath(os.path.dirname(__file__))


def analysis_dataset(txt):
    """
    解析dataset.txt
    """
    assert os.path.exists(txt), "错误: 文件不存在"
    imgs_list = TXT_Tools.read_lines(txt, split_flag=",")
    dataset = {
        "train": {"imgs": [], "labels": []},
        "val": {"imgs": [], "labels": []},
        "test": {"imgs": [], "labels": []},
    }
    labels = set()
    for path, label, mode in imgs_list:
        assert mode in ["train", "val", "test"]
        labels.add(label)
        dataset[mode]["imgs"].append(path)
        dataset[mode]["labels"].append(label)
    labels = list(labels)
    labels.sort()

    index = list(range(0, len(labels)))
    labels_dict = dict(zip(index, labels))

    dataset["labels"] = labels
    dataset["labels_dict"] = labels_dict

    return dataset


def vis_cam(model, img_tensor, pool_name="global_pool"):
    """
    可视化注意力图

    img_tensor(tensor): shape[B,C,H,W]

    pool_name(str): 可视化特征图的网络位置的名称。
        通常选取卷积网络最后输出的特征图  (卷积网络->全局池化->分类网络)
        默认timm库的全局池化名称为"global_pool",自定义模型需自行确定

    更多可视化算法,请访问 https://github.com/jacobgil/pytorch-grad-cam
    """
    from pytorch_grad_cam import GradCAM

    cam_algorithm = GradCAM
    modules_list = []
    for name, module in model.named_modules():
        if pool_name in name:  # 定位到全局池化层
            break
        modules_list.append(module)
    target_layers = [modules_list[-1]]  # 全局池化层的前一层

    # 反归一化、RGB->BGR、[B,C,H,W] -> [B,H,W,C]
    bgr_img = Img_Tools.tensor2img(img_tensor.cpu(), BCHW2BHWC=True)
    bgr_img = bgr_img.squeeze(0).numpy()

    try:
        with cam_algorithm(model=model, target_layers=target_layers) as cam:
            cam.batch_size = 32
            grayscale_cam = cam(
                input_tensor=img_tensor,
                targets=None,  # 默认基于模型预测最高分值的类别可视化
                aug_smooth=True,  # 平滑策略1
                eigen_smooth=True,  # 平滑策略2
            )
            grayscale_cam = grayscale_cam[0, :]

            cam_image = show_cam_on_image(bgr_img, grayscale_cam, use_rgb=False)
        return cam_image
    except:
        print("错误: 请尝试确认 当前模型的全局池化层名称，并赋值pool_name")
        sys.exit()


def convert_vis(imgs, category, per_nums=4):
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
    imgs_list = [Img_Tools.tensor2img(imgs) for imgs in imgs_list]
    # 拼成网格图CHW
    imgs_list = [torchvision.utils.make_grid(line) for line in imgs_list]
    return imgs_list
