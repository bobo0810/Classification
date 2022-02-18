import sys
import os
import torch
import time
import yaml
from DataSets.dataset import TestSet
from Models.Backbone import Backbone
from Utils.tools import eval_confusion_matrix

cur_path = os.path.abspath(os.path.dirname(__file__))
if __name__ == "__main__":
    device = torch.device("cuda:0")  # 'cuda:0'
    file = open(cur_path + "/Config/test.yaml", "r")
    cfg = yaml.load(file, Loader=yaml.FullLoader)

    model = Backbone(
        cfg["Models"]["backbone"],
        class_nums=len(cfg["DataSet"]["category"]),
        checkpoint=cfg["Models"]["checkpoint"],
    )
    model.to(device)
    model.eval()

    test_dataloader = torch.utils.data.DataLoader(
        dataset=TestSet(cfg["DataSet"]),
        batch_size=cfg["Models"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 测试
    acc = eval_confusion_matrix(model, test_dataloader, device)
    print("acc is " + str(acc.Overall_ACC))
    acc.print_matrix()
    acc.print_normalized_matrix()
