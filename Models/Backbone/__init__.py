import torch.nn as nn
import timm
import torch
from collections import OrderedDict


class Backbone(nn.Module):
    """
    主干网络入口
    """

    def __init__(self, model_name, class_nums, checkpoint=None):
        """
        model_name: 模型名称
        class_nums: 类别数
        """
        super(Backbone, self).__init__()
        self.model = self.init_model(model_name, class_nums)
        if checkpoint:
            print("checkpoint load {}".format(checkpoint))
            self.model = self.load_checkpoint(self.model, checkpoint)

    def forward(self, imgs):
        return self.model(imgs)

    @staticmethod
    def init_model(model_name, class_nums):
        """
        初始化主干网络

        加载优先级：Timm模型库 > 本地模型

        Timm模型库: https://github.com/rwightman/pytorch-image-models
        """
        model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=class_nums,
        )
        return model

    @staticmethod
    def load_checkpoint(model, checkpoint):
        """
        加载预训练权重
        """
        state_dict = torch.load(checkpoint, map_location=torch.device("cpu"))
        model_dict = model.state_dict()  # 模型参数
        new_state_dict = OrderedDict()
        # 遍历预训练参数
        for k, v in state_dict.items():
            name = k
            # GPU并行的预训练参数,需移除'module.'
            if "module." in name:
                name = name[7:]
            if "model." in name:
                name = name[6:]
            if name in model_dict:
                new_state_dict[name] = v
            else:
                print("!" * 6, "model parameter  mismatch:" + name, "!" * 6)
        model_dict.update(new_state_dict)
        model.load_state_dict(model_dict, strict=True)
        return model


# def test():
#     """测试示例"""
#     import torch

#     # model_names = timm.list_models(pretrained=True)  # 列出具有预训练权重的模型
#     model = Backbone("resnest50d", class_nums=6)
#     imgs = torch.ones((8, 3, 224, 224))
#     predict = model(imgs)
#     print(predict.shape)
