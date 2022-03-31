import torch
import math
import random
import torchmetrics
import time
import os
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
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


def vis_cam(model, img_tensor, img_path, target_layers):
    """
    可视化注意力图

    img_tensor: shape[B,C,H,W]
    """
    # rgb_img = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [1,C,H,W]->[H,W,C]
    bgr_img = cv2.imread(img_path, 1)
    bgr_img = cv2.resize(bgr_img, (224, 224), interpolation=cv2.INTER_CUBIC)
    bgr_img = np.float32(bgr_img) / 255  # 归一化

    with GradCAM(model=model, target_layers=target_layers) as cam:
        cam.batch_size = 32
        grayscale_cam = cam(
            input_tensor=img_tensor,  # 输入tensor
            targets=None,  # 默认按模型预测最高分值的类别 可视化
            aug_smooth=True,  # 平滑策略1
            eigen_smooth=True,  # 平滑策略2
        )
        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(bgr_img, grayscale_cam, use_rgb=False)
    return cam_image
