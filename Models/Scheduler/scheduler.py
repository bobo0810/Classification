from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR, MultiStepWarmupLR

# CosineLR     余弦预热学习率调度器
# MultistepLR  多步预热学习率调度器


def CosineLR(epochs, optimizer):
    """
    余弦预热-学习率调度器
    """
    return CosineAnnealingWarmupLR(optimizer, epochs, warmup_steps=int(epochs * 0.1))


def MultistepLR(epochs, optimizer):
    """
    多步预热-学习率调度器
    """
    return MultiStepWarmupLR(
            optimizer,
            epochs,
            warmup_steps=int(epochs * 0.1),  # 预热轮数
            milestones=[int(epochs * 0.7), int(epochs * 0.9)],  # 总轮数的70%、90%时调整学习率
            gamma=0.1,  # 学习率下降的倍数
        )
