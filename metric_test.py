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
    parser.add_argument("--width", help="图像宽", default=224)
    parser.add_argument("--height", help="图像高", default=224)
    parser.add_argument("--batch", type=int, help="推理batch", default=8)
    # 参数
    parser.add_argument("--txt", help="数据集路径", default=cur_path + "/Config/dataset.txt")
    parser.add_argument("--process", help="图像预处理", default="ImageNet")
    parser.add_argument("--mirror", help="融合镜像特征", default=False)
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
        model,
        dataset,
        [args.height, args.width],
        args.process,
        args.batch,
        "test",
        args.mirror,
    )
    print(result)
