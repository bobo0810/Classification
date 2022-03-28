import torch.nn as nn
import timm
import torch
from collections import OrderedDict
from .mynet import mynet  # 触发注册器


class create_backbone(nn.Module):
    """
    主干网络入口
    """

    def __init__(self, model_name, num_classes, checkpoint=None):
        """
        model_name: 模型名称,即yaml文件backbone属性值
        num_classes: 类别数
        """
        super(create_backbone, self).__init__()
        self.model = self.init_model(model_name, num_classes)

    def forward(self, imgs):
        return self.model(imgs)

    def init_model(self, model_name, num_classes):
        """
        初始化主干网络
        """
        return timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes,
        )
