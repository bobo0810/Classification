import torch
import math
import random
import torchmetrics
import time
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter

cur_path = os.path.abspath(os.path.dirname(__file__))


def get_labels(path):
    """
    读取label.txt，获取类别字典

    eg: {'dog': 0, 'cat': 1}
    """
    assert os.path.exists(path), "Warn: %s does not exist" % path
    labels = open(path, "r").readlines()
    labels = [label.strip() for label in labels if label != "\n"]
    index = list(range(0, len(labels)))

    return dict(zip(labels, index))


def init_env(cfg):
    """
    初始化训练环境
    """
    # 固定随机种子
    seed = 227
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 设置CUDA
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    # 创建日志路径
    exp_path = (
        os.path.dirname(cur_path)
        + "/ExpLog/"
        + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        + "/"
    )
    tb_path, checkpoint_path = [exp_path + "tb_log/", exp_path + "checkpoint/"]
    os.makedirs(tb_path)
    os.makedirs(checkpoint_path)

    # 初始化TensorBoard
    tb_writer = SummaryWriter(tb_path)
    tb_writer.add_text("Config", str(cfg))
    print("*" * 28)
    print("TensorBoard | Checkpoint save to ", exp_path, "\n")
    return tb_writer, checkpoint_path


@torch.no_grad()
def eval_metric(model, data_loader, device):
    """
    评估指标

    acc: 准确率
    cm:  混淆矩阵
    """
    scores_list, preds_list, labels_list = [], [], []
    for batch_idx, (imgs, labels, _) in enumerate(data_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        scores = model(imgs)
        scores = torch.nn.functional.softmax(scores, dim=1)
        preds = torch.argmax(scores, dim=1)

        scores_list.append(scores)
        preds_list.append(preds)
        labels_list.append(labels)

    scores_tensor = torch.cat(scores_list, dim=0)  # [imgs_nums,class_nums]
    preds_tensor = torch.cat(preds_list, dim=0)  # [imgs_nums]
    labels_tensor = torch.cat(labels_list, dim=0)  # [imgs_nums]

    # 统计
    metric_acc = torchmetrics.Accuracy().to(device)
    metric_cm = torchmetrics.ConfusionMatrix(
        num_classes=len(data_loader.dataset.labels)
    ).to(device)
    acc = metric_acc(scores_tensor, labels_tensor)
    cm = metric_cm(preds_tensor, labels_tensor)
    return acc, cm
