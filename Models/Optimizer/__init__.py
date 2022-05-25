import torch
import torch.nn as nn
import timm
import timm.optim
# 当前支持
optimizer_list = [
    "sgd",
    "adam",
    "lamb",
]


def create_optimizer(params, opt_name, lr):
    """
    优化器入口

    params: 模型或参数
    opt_name: 优化器名称
    lr: 学习率
    """
    assert opt_name in optimizer_list, "NotImplementedError"

    optimizer = timm.optim.create_optimizer_v2(
        params, opt=opt_name, lr=lr, weight_decay=0.0005, momentum=0.9
    )
    return optimizer
