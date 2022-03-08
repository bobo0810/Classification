import sys
import os
import argparse
import random
import glob
import cv2
import numpy as np
from tqdm import tqdm
import shutil
from ToolsLib.Img_Tools import Img_Tools
from ToolsLib.TXT_Tools import TXT_Tools


random.seed(224)
cur_path = os.path.abspath(os.path.dirname(__file__))


def verifyImgs(imgs_path):
    """
    验证图像可读性
    imgs_path目录下
    """
    imgs_list = glob.glob(os.path.join(imgs_path, "*/*"))  # 所有图片列表
    error_list = Img_Tools.verify_integrity(imgs_list)
    if len(error_list) > 0:
        print("Strongly recommended: Delete the wrong image")
        print(error_list)


def split(imgs_path, ratio, txt_path, prefix=None):
    """
    划分数据集
    每类均按比例分配train、test

    txt_path: train/test.txt保存路径
    prefix: 若存在，则图片路径去掉前缀
    """

    class_list = glob.glob(os.path.join(imgs_path, "*"))  # 所有类别
    train_list, test_list = [], []

    # 每类均按比例分配
    for class_path in class_list:
        imgs_list = glob.glob(os.path.join(class_path, "*"))
        if prefix:
            prefix = prefix + "/" if prefix[-1] != "/" else prefix
            imgs_list = [line.replace(prefix, "") for line in imgs_list]  # eg: 类别名/图片名
        random.shuffle(imgs_list)

        train_list.extend(imgs_list[: int(len(imgs_list) * ratio)])
        test_list.extend(imgs_list[int(len(imgs_list) * ratio) :])

    # 保存
    TXT_Tools.write_lines(train_list, os.path.join(txt_path, "train.txt"))
    TXT_Tools.write_lines(test_list, os.path.join(txt_path, "test.txt"))


if __name__ == "__main__":
    """
    读取数据集根路径，划分训练集、测试集
    """
    parser = argparse.ArgumentParser(description="划分数据集")
    parser.add_argument(
        "--ImgsPath", required=True, help="数据集根路径  eg: /home/xxx/CatDog/"
    )
    parser.add_argument("--Ratio", type=float, default=0.8, help="train:test=0.8:0.2")
    parser.add_argument("--Verify", action="store_true", help="验证图像完整性(耗时)")
    parser.add_argument(
        "--TxtPath", default=cur_path + "/../Config/", help="train/test.txt保存路径"
    )

    args = parser.parse_args()

    if args.Verify:
        verifyImgs(args.ImgsPath)

    split(
        imgs_path=args.ImgsPath,
        ratio=args.Ratio,
        txt_path=args.TxtPath,
        prefix=args.ImgsPath,
    )
