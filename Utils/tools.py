import torch
import math
import random
import cv2
from pycm import ConfusionMatrix  # 统计混淆矩阵


def init_seed(seed: int):
    """
    设置随机种子
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 非确定性算法
    torch.backends.cudnn.enabled = True
    # 自动为每个卷积层搜索最适合的实现算法，加速训练
    # 适用场景:网络结构固定（不是动态变化的），输入形状（batch size，img shape，channel）不变
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


@torch.no_grad()
def eval_model(model, data_loader, device):
    """
    评估模型（验证集）
    """
    img_nums = 0  # 验证集总数量
    correct_nums = 0  # 统计预测正确的数量
    for batch_idx, (imgs, labels) in enumerate(data_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        scores = model(imgs)
        scores = torch.nn.functional.softmax(scores, dim=1)
        _, indices = torch.sort(scores, dim=1, descending=True)
        pred_label = indices[:, 0]

        for i in range(len(imgs)):
            if labels[i] == pred_label[i]:
                correct_nums += 1
        img_nums += len(imgs)
    acc = correct_nums / img_nums
    return acc  # ACC准确率


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
