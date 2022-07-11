# ======数据集============
Size = [224, 224]  # 图像尺寸
Sampler = "batch_balance"  # 采样策略
Txt = "./Config/dataset.txt"  # 数据集路径
Process = "ImageNet"  # 图像预处理策略

# ======模型============
Optimizer = "SGD"  # 优化器

# 常规分类
Backbone = "resnet18"  # 主干网络(名称or权重路径)
Loss = "CrossEntropy"  # 损失函数

# 度量学习
# Backbone = "resnet18"  # 主干网络(名称or权重路径)
# Loss = "ArcFace"       # 损失函数|分类器(名称or权重路径)
# Feature_dim = 128

# ======训练============
LR = 0.01  # 学习率
Batch = 64  # 批次
Epochs = 80  # 总轮数
Scheduler = "CosineLR"  # 学习率调度器

# ======分布式============
from colossalai.amp import AMP_TYPE

# 混合精度
fp16 = dict(mode=AMP_TYPE.TORCH)

# 梯度积累
# gradient_accumulation = 4

# 梯度裁剪
# clip_grad_norm = 1.0

# 流水线并行,Tensor并行 请查阅文档https://www.colossalai.org/zh-Hans/docs/features/pipeline_parallel
