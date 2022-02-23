from torchvision.models.alexnet import alexnet


def MyNet(pretrained=True, num_classes=1000):
    """
    自定义backbone

    TorchHub模型库: https://pytorch.org/hub/research-models
    """
    return alexnet(pretrained=pretrained, num_classes=num_classes)
