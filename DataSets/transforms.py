from torchvision import transforms

# ImageNet均值、方差
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


def ImgTransforms(mode, img_shape=(224, 224)):
    """
    数据增广入口
    """
    assert mode in ["train", "val", "test"]

    # =========================训练集========================================
    if mode == "train":
        img_transforms = transforms.Compose(
            [
                transforms.Resize(img_shape),
                # 翻转
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                # 旋转
                transforms.RandomChoice(
                    [
                        # 在 (-a, a) 之间随机选择
                        transforms.RandomRotation(30),
                        transforms.RandomRotation(60),
                        transforms.RandomRotation(90),
                    ]
                ),
                # 颜色
                transforms.RandomChoice(
                    [
                        transforms.ColorJitter(brightness=0.5),
                        transforms.ColorJitter(contrast=0.5),
                        transforms.ColorJitter(saturation=0.5),
                        transforms.ColorJitter(hue=0.5),
                        transforms.ColorJitter(
                            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                        ),
                        transforms.ColorJitter(
                            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3
                        ),
                        transforms.ColorJitter(
                            brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
                        ),
                    ]
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                transforms.RandomErasing(),  # 遮挡
            ]
        )

    # =========================验证集/测试集==================================
    else:
        img_transforms = transforms.Compose(
            [
                transforms.Resize(img_shape),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    return img_transforms
