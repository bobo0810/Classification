import torch
import torch.nn as nn
from timm.loss import LabelSmoothingCrossEntropy
from pytorch_metric_learning import losses

# =================================分类任务损失===============================================


def CrossEntropy():
    """
    交叉熵损失函数
    """
    return nn.CrossEntropyLoss()


def LabelSmooth():
    """
    标签平滑的交叉熵损失函数
    """
    return LabelSmoothingCrossEntropy()


# ======================================度量学习损失==========================================


def CosFace(embedding_size, num_classes):
    """
    CosFace分类器

    embedding_size: 特征维度
    num_classes: 类别数
    """
    return losses.CosFaceLoss(num_classes=num_classes, embedding_size=embedding_size)


def ArcFace(embedding_size, num_classes):
    """
    ArcFace分类器

    embedding_size: 特征维度
    num_classes: 类别数
    """
    return losses.ArcFaceLoss(num_classes=num_classes, embedding_size=embedding_size)


def SubCenterArcFace(embedding_size, num_classes):
    """
    SubCenterArcFace分类器

    embedding_size: 特征维度
    num_classes: 类别数
    """
    return losses.SubCenterArcFaceLoss(num_classes=num_classes, embedding_size=embedding_size)


def CircleLoss(embedding_size, num_classes):
    """
    CircleLoss分类器
    """
    return losses.CircleLoss()
