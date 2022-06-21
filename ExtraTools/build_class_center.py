"""
获取训练集的类中心
"""
import sys
import glob
import os

rootpath = os.path.abspath(os.path.dirname(__file__)) + "/.."
sys.path.append(rootpath)
sys.path.extend(glob.glob(rootpath + "/*"))
from Utils.tools import analysis_dataset
from DataSets import create_datasets, create_dataloader
import torch
import numpy as np
from tqdm import tqdm
import argparse


@torch.no_grad()
def get_class_center(dataloader, model):
    """
    获取特征
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    labels_dict = {}
    for batch_idx, (imgs, labels) in enumerate(tqdm(dataloader)):
        imgs = imgs.to(device)
        features = model(imgs)

        for i in range(len(labels)):
            label = labels[i].item()
            if not label in labels_dict.keys():
                labels_dict[label] = []
            labels_dict[label].append(features[i])  # labels指 类别ID
    # 计算各类的特征均值，即类中心
    center_dict = {}
    for label, features in labels_dict.items():
        features = torch.stack(features)
        center_feature = torch.mean(features, dim=0)  # [fetaure_dim]
        center_dict[label] = center_feature.to("cpu")
    # 按类别id排序
    index_list = list(range(len(center_dict)))
    class_center = [center_dict[index] for index in index_list]
    return torch.stack(class_center).numpy()


if __name__ == "__main__":
    print("build class center...")
    parser = argparse.ArgumentParser(description="构建训练集类中心")
    # 默认参数
    parser.add_argument("--size", type=str, help="图像宽高", default="112,112")
    parser.add_argument("--process", help="图像预处理", default="FaceCompare")
    parser.add_argument("--batch", type=int, help="推理batch", default=512)
    # 参数
    parser.add_argument("--txt_path", help="新数据集路径 eg:/home/dataset.txt", required=True)
    parser.add_argument("--weight_path", help="特征模型路径", required=True)
    parser.add_argument("--save_npy", help="类中心的保存路径  eg:/home/xx.npy", required=True)
    args = parser.parse_args()
    assert ".npy" in args.save_npy

    args.size = [int(line) for line in args.size.split(",")]

    # 构建训练集
    dataset = analysis_dataset(args.txt_path)
    train_set = create_datasets(
        dataset=dataset["train"], size=args.size, process=args.process
    )
    train_dataloader = create_dataloader(args.batch, train_set)

    # 加载模型
    model = torch.load(args.weight_path, map_location="cpu")

    # 类中心 [class_nums,feature_dim] 
    class_center = get_class_center(train_dataloader, model)

    # 保存
    np.save(args.save_npy, class_center)