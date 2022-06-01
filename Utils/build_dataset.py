import sys
import os
import argparse
import random
import glob
import cv2
import numpy as np
from tqdm import tqdm
import shutil
from bobotools.img_tools import Img_Tools
from bobotools.txt_tools import TXT_Tools

random.seed(227)
cur_path = os.path.abspath(os.path.dirname(__file__))


def verifyImgs(imgs_path):
    """
    验证图像可读性
    imgs_path目录下
    """
    imgs_list = glob.glob(os.path.join(imgs_path, "*/*"))  # 所有图片列表
    error_list = Img_Tools.verify_integrity(imgs_list)
    if len(error_list) > 0:
        print("强烈建议: 请删除下列错误图片")
        print(error_list)


def split(imgs_path, ratio, dataset_txt):
    """
    每类按比例,划分数据集

    imgs_path: 数据集根路径  eg: /home/xxx/CatDog/
    ratio: 训练集、验证集、测试集的比例  eg:[0.7,0.1,0.2]
    dataset_txt: dataset.txt保存路径
    """
    assert sum(ratio) == 1.0

    class_list = glob.glob(os.path.join(imgs_path, "*"))  # 所有类别
    train_list, val_list, test_list = [], [], []

    # 每类按比例划分
    for class_path in class_list:
        imgs_list = glob.glob(os.path.join(class_path, "*"))
        random.shuffle(imgs_list)

        train_index = int(len(imgs_list) * ratio[0])
        val_index = train_index + int(len(imgs_list) * ratio[1])

        train_list.extend(imgs_list[:train_index])
        val_list.extend(imgs_list[train_index:val_index])
        test_list.extend(imgs_list[val_index:])

    dataset_list = []

    def combin_dataset(imgs_list, type):
        for img_path in imgs_list:
            label = img_path.split("/")[-2]
            dataset_list.append(type+ "," + label + ","+ img_path)

    combin_dataset(train_list, "train")
    combin_dataset(val_list, "val")
    combin_dataset(test_list, "test")

    TXT_Tools.write_lines(dataset_list, dataset_txt)


if __name__ == "__main__":
    """
    读取数据集,划分训练集、验证集、测试集。

    txt格式为[类型,类别名,图片路径]
    - 类型:  train:训练集  val:验证集  test:测试集
    - 类别名: cat dog
    - 路径: 绝对路径or相对路径
    """
    parser = argparse.ArgumentParser(description="划分数据集")
    parser.add_argument(
        "--ImgsPath", required=True, help="数据集根路径  eg: /home/xxx/CatDog/"
    )
    parser.add_argument("--Ratio", default=[0.7, 0.1, 0.2], help="训练集:验证集:测试集的各类别划分比例")
    parser.add_argument("--Verify", action="store_true", help="验证图像完整性(耗时,可选)")
    parser.add_argument(
        "--TxtPath",
        default=cur_path + "/../Config/dataset.txt",
        help="dataset.txt保存路径",
    )
    args = parser.parse_args()

    if args.Verify:
        verifyImgs(args.ImgsPath)
    split(args.ImgsPath, args.Ratio, args.TxtPath)
