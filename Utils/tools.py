import torch
import math
import random
from pycm import ConfusionMatrix
import time
import os
import cv2
import sys
from pytorch_metric_learning import losses, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
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


def get_category(path):
    """
    读取label.txt，获取类别

    eg: ['dog','cat']
    """
    assert os.path.exists(path), "Warn: %s does not exist" % path
    labels = open(path, "r").readlines()
    labels = [label.strip() for label in labels if label != "\n"]
    return labels


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
    return tb_writer, checkpoint_path + cfg["Models"]["backbone"]


@torch.no_grad()
def eval_model(model, data_loader):
    """
    常规分类：评估指标
    """
    device = next(model.parameters()).device

    scores_list, preds_list, labels_list = [], [], []
    for batch_idx, (imgs, labels) in enumerate(data_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        scores = model(imgs)
        scores = torch.nn.functional.softmax(scores, dim=1)
        preds = torch.argmax(scores, dim=1)

        preds_list.append(preds)
        labels_list.append(labels)

    preds_list = torch.cat(preds_list, dim=0).cpu().numpy() 
    labels_list = torch.cat(labels_list, dim=0).cpu().numpy()

    # 统计
    return ConfusionMatrix(labels_list, preds_list)


@torch.no_grad()
def eval_metric_model(model, train_set, val_set):
    """
    度量学习：评估指标
    """
    tester = testers.BaseTester(batch_size=64, dataloader_num_workers=4)
    train_embeddings, train_labels = tester.get_all_embeddings(train_set, model)
    test_embeddings, test_labels = tester.get_all_embeddings(val_set, model)
    train_labels, test_labels = train_labels.squeeze(1), test_labels.squeeze(1)

    accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)
    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings, train_embeddings, test_labels, train_labels, False
    )
    return accuracies["precision_at_1"]


def tensor2img(tensor, BCHW2BHWC=False):
    """
    Tenso恢复为图像，用于可视化
    反归一化、RGB->BGR

    tensor: Tensor,形状[B,C,H,W]
    BCHW2BHWC: (可选)是否交换Tensor维度

    返回值
    imgs: Tensor,形状[B,C,H,W]
    """
    B, C, H, W = tensor.shape
    # ImageNet均值方差
    t_mean = torch.FloatTensor((0.485, 0.456, 0.406)).view(C, 1, 1).expand(3, H, W)
    t_std = torch.FloatTensor((0.229, 0.224, 0.225)).view(C, 1, 1).expand(3, H, W)

    tensor = tensor * t_std.to(tensor) + t_mean.to(tensor)  # 反归一化
    tensor = tensor[:, [2, 1, 0], :, :]  # RGB->BGR
    if BCHW2BHWC:
        tensor = tensor.permute(0, 2, 3, 1)
    return tensor


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

    # 反归一化、RGB->BGR、[B,C,H,W] -> [B,H,W,C]
    bgr_img = tensor2img(img_tensor.cpu(), BCHW2BHWC=True)
    bgr_img = bgr_img.squeeze(0).numpy()

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
