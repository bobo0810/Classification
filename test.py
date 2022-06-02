import sys
import os
import torch
import time
from DataSets import create_datasets, create_dataloader
from Utils.eval import eval_model
from Utils.tools import analysis_dataset
import argparse
import matplotlib.pyplot as plt

cur_path = os.path.abspath(os.path.dirname(__file__))
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试-分类任务")
    # 默认参数
    parser.add_argument("--size", help="图像尺寸", default=[224, 224])
    parser.add_argument("--batch", type=int, help="推理batch", default=8)
    # 参数
    parser.add_argument("--txt", help="数据集路径", default=cur_path + "/Config/dataset.txt")
    parser.add_argument("--process", help="图像预处理", default="ImageNet")
    parser.add_argument("--weights", help="模型权重", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 直接加载model,而非model.state_dict
    model = torch.load(args.weights, map_location="cpu")
    model.to(device)
    model.eval()
    print(f"model info is {model.info}")

    # 常规分类
    assert model.info["task"] == "class", "警告: 该模型不是分类任务模型"

    # 数据集
    dataset = analysis_dataset(args.txt)
    test_set = create_datasets(
        dataset=dataset["test"], size=args.size, process=args.process
    )
    test_dataloader = create_dataloader(args.batch,test_set)
    

    # 统计指标
    cm = eval_model(model, test_dataloader)
    cm.relabel(mapping=dict(zip(list(range(0, len(dataset["all_labels"]))), dataset["all_labels"])))    
    print("Overall ACC is %.3f \n" % cm.Overall_ACC)

    # 可视化混淆矩阵
    cm.plot(cmap=plt.cm.Reds, normalized=True, number_label=True, plot_lib="seaborn")
    plt.savefig(cur_path + "/matrix.jpg")
    print("matrix save in ", cur_path + "/matrix.jpg \n")

    # 输出全部指标
    cm.print_normalized_matrix()
    print(cm)
