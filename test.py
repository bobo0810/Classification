import sys
import os
import torch
import time
import yaml
from DataSets import create_dataloader
from Utils.tools import eval_metric, get_labels
import argparse

cur_path = os.path.abspath(os.path.dirname(__file__))
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument("--yaml", help="测试配置", default=cur_path + "/Config/test.yaml")
    parser.add_argument("--txt", help="测试集路径", default=cur_path + "/Config/test.txt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    file = open(args.yaml, "r")
    cfg = yaml.load(file, Loader=yaml.FullLoader)
    cfg["DataSet"]["txt"] = args.txt
    cfg["DataSet"]["labels"] = get_labels(
        path=os.path.dirname(args.txt) + "/labels.txt"
    )

    assert (
        cfg["Models"]["checkpoint"] != None
    ), "Warn: test.yaml checkpoint should not be None"

    # 直接加载model,而非model.state_dict
    model = torch.load(cfg["Models"]["checkpoint"], map_location="cpu")
    try:
        model = model.module if model.module else model
    except:
        pass
    model.to(device)
    model.eval()

    test_dataloader = create_dataloader(cfg["DataSet"], mode="test")

    # 输出ACC及混淆矩阵
    acc, cm = eval_metric(model, test_dataloader, device)
    print("accuracy is %.3f \n" % acc)
    print("confusion matrix is  \n", cm)
