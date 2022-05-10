import sys
import os
import torch
import time
from DataSets import create_dataloader
from DataSets.dataset import create_datasets
from Utils.tools import analysis_dataset, eval_model, eval_metric_model
import argparse
import matplotlib.pyplot as plt

cur_path = os.path.abspath(os.path.dirname(__file__))
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试")
    # 默认参数
    parser.add_argument("--size", type=list,help="图像尺寸", default=[224,224])
    parser.add_argument("--batch", type=int,help="推理batch", default=8)
    # 参数
    parser.add_argument("--txt", help="测试集路径", default=cur_path + "/Config/dataset.txt")
    parser.add_argument("--checkpoint",help="测试集路径", required=True)
    args = parser.parse_args()
    dataset_params={'size':args.size,'batch':args.batch,'txt':args.txt}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 直接加载model,而非model.state_dict
    model = torch.load(args.checkpoint, map_location="cpu")
    while hasattr(model, "module"):
        model = model.module
    model.to(device)
    model.eval()
    TASK = "metric" if hasattr(model, "embedding_size") else "class"

    if TASK == "class":  # 常规分类
        # 数据集
        test_dataloader = create_dataloader(dataset_params, mode="test")
        labels_list = analysis_dataset(args.txt)["labels_dict"]

        # 统计指标
        cm = eval_model(model, test_dataloader)
        cm.relabel(mapping=labels_list)
        print("Overall ACC is %.3f \n" % cm.Overall_ACC)

        # 可视化混淆矩阵
        cm.plot(
            cmap=plt.cm.Reds, normalized=True, number_label=True, plot_lib="seaborn"
        )
        plt.savefig(cur_path + "/matrix.jpg")
        print("matrix save in ", cur_path + "/matrix.jpg \n")

        # 输出全部指标
        cm.print_normalized_matrix()
        print(cm)
    elif TASK == "metric":  # 度量学习
        # 数据集
        train_set = create_datasets(dataset_params, mode="train")
        test_set = create_datasets(dataset_params, mode="test")

        # 统计精确率
        precision = eval_metric_model(model, train_set, test_set)
        print("precision is %.3f \n" % precision)
