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

    训练集仅支持 [类型,类别名,图片1]

    验证集、测试集 支持 [类型,类别名,图片1] 或  [类型,是否为同类,图片1,图片2]样本对

    度量学习 用于对比两个图片的相似度。

    eg:
    [
        train, dog,  img1.jpg

        val,   true,  img1.jpg,  img2.jpg
        val,   false, img3.jpg,  img4.jpg

        test,  true,  img5.jpg,  img6.jpg
        test,  false, img7.jpg,  img8.jpg
    ]

    return
    [类型,类别名,图片1]          ->    "imgs": [],  "labels": [], "all_labels": []
    [类型,是否为同类,图片1,图片2] ->    "imgs": [],  "positive_pairs"=[] ,"negative_pairs" = [], "all_labels": []
    """
    assert os.path.exists(txt), "错误: 文件不存在"
    txt_list = TXT_Tools.read_lines(txt, split_flag=",")

    all_labels = set()  # 所有类别
    train_data = {"imgs": [], "labels": []}  # 训练集
    val_data = {"imgs": []}  # 验证集
    test_data = {"imgs": []}  # 测试集
    for line in txt_list:
        assert line[0] in ["train", "val", "test"]
        if line[0] == "train":
            # [类型,类别名,图像路径]格式
            all_labels.add(line[1])
            train_data["labels"].append(line[1])
            train_data["imgs"].append(line[2])

        elif line[0] == "val":
            if len(line) == 3:
                # [类型,类别名,图像路径]格式
                if not "labels" in val_data.keys():
                    val_data["labels"] = []

                val_data["labels"].append(line[1])
                val_data["imgs"].append(line[2])
            else:
                # [类型,是否为同类,图片1,图片2]样本对格式
                assert line[1] in ["true", "false"]
                if not "positive_pairs" in val_data.keys():
                    val_data["positive_pairs"] = []
                if not "negative_pairs" in val_data.keys():
                    val_data["negative_pairs"] = []
                val_data["imgs"].extend([line[2], line[3]])
                if line[1] == "true":
                    val_data["positive_pairs"].append([line[2], line[3]])
                else:
                    val_data["negative_pairs"].append([line[2], line[3]])
        elif line[0] == "test":
            if len(line) == 3:
                # [类型,类别名,图像路径]格式
                if not "labels" in test_data.keys():
                    test_data["labels"] = []

                test_data["labels"].append(line[1])
                test_data["imgs"].append(line[2])
            else:
                # [类型,是否为同类,图片1,图片2]样本对格式
                assert line[1] in ["true", "false"]
                if not "positive_pairs" in test_data.keys():
                    test_data["positive_pairs"] = []
                if not "negative_pairs" in test_data.keys():
                    test_data["negative_pairs"] = []

                test_data["imgs"].extend([line[2], line[3]])
                if line[1] == "true":
                    test_data["positive_pairs"].append([line[2], line[3]])
                else:
                    test_data["negative_pairs"].append([line[2], line[3]])

    # 提取训练集的所有类别
    all_labels = list(all_labels)
    all_labels.sort()

    train_data["all_labels"] = all_labels
    if "positive_pairs" in val_data.keys():
        val_data["imgs"] = list(set(val_data["imgs"]))
    else:
        val_data["all_labels"] = all_labels
    if "positive_pairs" in test_data.keys():
        test_data["imgs"] = list(set(test_data["imgs"]))
    else:
        test_data["all_labels"] = all_labels

    dataset = {
        "train": train_data,
        "val": val_data,
        "test": test_data,
        "all_labels": all_labels,
    }
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


@torch.no_grad()
def get_feature(
    dataloader,
    model,
    device,
    use_mirror=False,
):
    """
    度量学习：提取特征
    """
    img_to_feature = {}
    for i, (img, img_path) in enumerate(dataloader):
        img = img.to(device)
        feature_npy = model(img).detach().cpu().numpy()
        if use_mirror:
            feature_npy += model(img.flip(-1)).detach().cpu().numpy()
        # 保存图像及对应特征
        for j in range(len(img_path)):
            img_to_feature[img_path[j]] = feature_npy[j]
    return img_to_feature


def get_score(img_to_feature, positive_pairs, negative_pairs):
    """
    度量学习：根据特征结果 和 正负样本对，计算余弦分数
    """

    # 保存 正样本对 每条记录的余弦相似度
    positive_score = []
    for img1, img2 in positive_pairs:
        feature_1 = img_to_feature[img1]
        feature_2 = img_to_feature[img2]
        # 计算 证件照和生活照的特征结果 的余弦相似度
        positive_score.append(
            np.inner(feature_1, feature_2)
            / np.power(np.sum(np.power(feature_1, 2)), 0.5)
            / np.power(np.sum(np.power(feature_2, 2)), 0.5)
        )

    # 保存 负样本对 每条记录的余弦相似度
    negative_score = []
    for img1, img2 in negative_pairs:
        feature_1 = img_to_feature[img1]
        feature_2 = img_to_feature[img2]
        negative_score.append(
            np.inner(feature_1, feature_2)
            / np.power(np.sum(np.power(feature_1, 2)), 0.5)
            / np.power(np.sum(np.power(feature_2, 2)), 0.5)
        )
    return positive_score, negative_score


def cal_index(positive_score, negative_score):
    """
    度量学习：统计 误识率下的通过率
    """
    P = len(positive_score)
    N = len(negative_score)
    score_cosin = positive_score
    score_cosin.extend(negative_score)
    label = [1] * P
    label.extend([0] * N)
    score = sorted(score_cosin, reverse=True)
    index = np.argsort(-np.array(score_cosin))
    label_sort = []
    for i in range(len(index)):
        label_sort.append(label[index[i]])

    TPR = []
    FPR = []

    # 万分之一、千分之一 误识率及对应通过率
    FPR_List = [0.0001, 0.001]
    TPR_List = [-1, -1]

    for idx in range(len(score)):
        FN = P - np.array(label_sort[0 : idx + 1]).sum()
        FP = idx + 1 - np.array(label_sort[0 : idx + 1]).sum()
        false_accept_rate = FP / N
        false_reject_rate = FN / P
        TPR.append(1 - false_reject_rate)
        FPR.append(false_accept_rate)
        if FPR[idx] > FPR_List[0] and TPR_List[0] == -1:
            TPR_List[0] = TPR[idx]
        if FPR[idx] > FPR_List[1] and TPR_List[1] == -1:
            TPR_List[1] = TPR[idx]
            break
    # 万分之一、千分之一 误识率下的通过率
    return dict(zip(FPR_List, TPR_List))
