import sys
import os
import torch
import time
from DataSets import create_datasets, create_dataloader
from Utils.eval import eval_metric_model
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
    print(f"extra info is {model.info}")

    # 度量学习
    assert model.info["task"] == "metric", "警告: 该模型不是度量学习模型"
    # 数据集
    train_set = create_datasets(
        txt=args.txt, mode="train", size=args.size, process=args.process
    )
    test_set = create_datasets(
        txt=args.txt, mode="test", size=args.size, process=args.process
    )
    # 统计精确率
    precision = eval_metric_model(model, train_set, test_set, args.batch)
    print("precision is %.3f \n" % precision)
