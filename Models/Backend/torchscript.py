import torch


class ScriptBackend:
    """
    TorchScript后端
    """

    def __init__(self):
        pass

    @staticmethod
    def convert(model, imgs, weights):
        """
        torch模型转为torchscript模型

        model: torch模型
        imgs: [B,C,H,W]Tensor
        weights: 权重保存路径
        """
        script_model = torch.jit.trace(model, imgs)
        torch.jit.save(script_model, weights)
        print("*" * 28)
        print("TorchScript export success, saved as %s" % weights)

    @staticmethod
    def infer(weights, imgs):
        """
        加载模型并推理

        weights(str): 权重路径
        img(tensor): [B,C,H,W]
        """
        script_model = torch.jit.load(weights)
        return script_model(imgs)
