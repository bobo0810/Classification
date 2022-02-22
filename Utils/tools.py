import torch
import math
import random
from pycm import ConfusionMatrix  # 统计混淆矩阵
import time
import os
from torch.utils.tensorboard import SummaryWriter

cur_path = os.path.abspath(os.path.dirname(__file__))


def init_env(cfg):
    """
    初始化训练环境
    """
    # 固定随机种子
    seed = 227
    random.seed(seed)
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
def eval_confusion_matrix(model, data_loader, device):
    """统计混淆矩阵"""
    pred_list, label_list = [], []
    for batch_idx, (imgs, labels) in enumerate(data_loader):
        imgs = imgs.to(device)
        scores = model(imgs)
        scores = torch.nn.functional.softmax(scores, dim=1)
        _, indices = torch.sort(scores, dim=1, descending=True)
        pred_label = indices[:, 0]
        for i in range(len(imgs)):
            pred_list.append(pred_label[i].cpu().item())
            label_list.append(labels[i].cpu().item())
    return ConfusionMatrix(actual_vector=label_list, predict_vector=pred_list)
