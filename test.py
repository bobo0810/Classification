import sys
import os
import torch
import time
import yaml
from DataSets import create_dataloader
from DataSets.dataset import create_datasets
from Utils.tools import analysis_dataset, eval_model, eval_metric_model
import argparse
import matplotlib
import matplotlib.pyplot as plt

cur_path = os.path.abspath(os.path.dirname(__file__))
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument("--yaml", help="测试配置", default=cur_path + "/Config/test.yaml")
    parser.add_argument("--txt", help="测试集路径", default=cur_path + "/Config/dataset.txt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = yaml.load(open(args.yaml, "r"), Loader=yaml.FullLoader)
    cfg["DataSet"]["txt"] = args.txt

    # 直接加载model,而非model.state_dict
    model = torch.load(cfg["Models"]["checkpoint"], map_location="cpu")
    while hasattr(model, "module"):
        model = model.module
    model.to(device)
    model.eval()
    TASK = "metric" if hasattr(model, "embedding_size") else "class"

    if TASK == "class":  # 常规分类
        # 数据集
        test_dataloader = create_dataloader(cfg["DataSet"], mode="test")
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
        train_set = create_datasets(cfg["DataSet"], mode="train")
        test_set = create_datasets(cfg["DataSet"], mode="test")

        # 统计精确率
        precision = eval_metric_model(model, train_set, test_set)
        print("precision is %.3f \n" % precision)
