import torch
import math
import random
import torchmetrics
import time
import os
import cv2
import sys
from pytorch_grad_cam import (
    GradCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    EigenGradCAM,
    LayerCAM,
    FullGrad,
)
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


def vis_cam(model, img_tensor, pool_name="global_pool", cam_algorithm=GradCAM):
    """
    可视化注意力图

    img_tensor(tensor): shape[B,C,H,W]

    pool_name(str): 可视化特征图的网络位置的名称。
        通常选取卷积网络最后输出的特征图  (卷积网络->全局池化->分类网络)
        默认timm库的全局池化名称为"global_pool",自定义模型需自行确定

    cam_algorithm: 可视化算法，包含:
        GradCAM, 默认
        ScoreCAM,
        GradCAMPlusPlus,
        AblationCAM,
        XGradCAM,
        EigenCAM,
        EigenGradCAM,
        LayerCAM,
        FullGrad,
    """
    modules_list = []
    for name, module in model.named_modules():
        if pool_name in name:  # 定位到全局池化层
            break
        modules_list.append(module)
    target_layers = [modules_list[-1]]  # 全局池化层的前一层

    # ImageNet均值、方差
    t_mean = torch.FloatTensor((0.485, 0.456, 0.406)).view(3, 1, 1).expand(3, 224, 224)
    t_std = torch.FloatTensor((0.229, 0.224, 0.225)).view(3, 1, 1).expand(3, 224, 224)

    # 1. [B,C,H,W]->[C,H,W] 2. 反归一化
    rgb_img = img_tensor.cpu().squeeze(0) * t_std + t_mean
    # 1. RGB->BGR 2. [C,H,W] -> [H,W,C]
    bgr_img = rgb_img[[2, 1, 0], :, :].permute(1, 2, 0).numpy()

    try:
        with cam_algorithm(model=model, target_layers=target_layers) as cam:
            cam.batch_size = 32
            grayscale_cam = cam(
                input_tensor=img_tensor,
                targets=None,  # 默认基于模型预测最高分值的类别可视化
                aug_smooth=True,  # 平滑策略1
                eigen_smooth=True,  # 平滑策略2
            )
            grayscale_cam = grayscale_cam[0, :]

            cam_image = show_cam_on_image(bgr_img, grayscale_cam, use_rgb=False)
        return cam_image
    except:
        print("错误: 请尝试确认 当前模型的全局池化层名称，并赋值pool_name")
        sys.exit()
