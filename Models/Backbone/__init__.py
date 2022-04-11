import torch.nn as nn
import timm
import torch
from collections import OrderedDict
from .mynet import mynet  # 注册自定义分类模型
from .mynet_metric import mynet_metric  # 注册自定义特征提取模型


def create_backbone(model_name, num_classes):
    """
    主干网络入口

    model_name: 模型名称,即yaml文件backbone属性值
    num_classes: 类别数
    """
    return timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes,
    )
