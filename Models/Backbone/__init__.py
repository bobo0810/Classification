import torch.nn as nn
import timm
import torch
from collections import OrderedDict
from .mynet import MyNet


class create_backbone(nn.Module):
    """
    主干网络入口
    """

    def __init__(self, model_name, num_classes, checkpoint=None):
        """
        model_name: 模型名称,即yaml文件backbone属性值
        num_classes: 类别数,即yaml文件category属性值的长度
        """
        super(create_backbone, self).__init__()
        self.model = self.init_model(model_name, num_classes)
        if checkpoint:
            print("checkpoint load {}".format(checkpoint))
            self.model = self.load_checkpoint(self.model, checkpoint)

    def forward(self, imgs):
        return self.model(imgs)

    @staticmethod
    def init_model(model_name, num_classes):
        """
        初始化主干网络
        优先级：自定义>timm库
        """
        if model_name == "MyNet":
            model = MyNet(pretrained=True, num_classes=num_classes)
        else:
            print("*" * 28)
            print(model_name, " come from timm\n")
            model = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=num_classes,
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
            # 移除前缀
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
