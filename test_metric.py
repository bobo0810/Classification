import sys
import os
import torch
import time
from DataSets import create_datasets, create_dataloader
from Utils.eval import eval_metric_model
from Utils.tools import analysis_dataset
import argparse
import matplotlib.pyplot as plt

cur_path = os.path.abspath(os.path.dirname(__file__))
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试-度量学习")
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

    # 度量学习
    assert model.info["task"] == "metric", "警告: 该模型不是度量学习模型"
    # 数据集
    dataset = analysis_dataset(args.txt)
    # 统计
    result = eval_metric_model(
        model, dataset, args.size, args.process, args.batch, mode="test"
    )
    if isinstance(result,dict):
        # 误识率下通过率
        print("FPR=1e-4，TPR= %.4f \n" % result[0.0001])
        print("FPR=1e-3，TPR= %.4f \n" % result[0.001])
    else:
        # 精准率
        print("precision is %.3f \n" % result)
