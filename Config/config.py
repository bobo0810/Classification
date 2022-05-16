# ======数据集============
Size = [224, 224]  # 图像尺寸
Sampler = "batch_balance"  # 采样策略
Txt = "./Config/dataset.txt"  # 数据集路径

# ======模型============
Optimizer = "sgd"  # 优化器

# 常规分类
# Backbone="mynet"   # 主干网络
# Loss="cross_entropy"    # 损失函数

# 度量学习
Backbone = "mynet_metric"
Loss = "arcface"
Feature_dim = 128

# ======训练============
LR = 0.01  # 学习率
Batch = 64  # 批次
Epochs = 80  # 总轮数


# ======分布式============
from colossalai.amp import AMP_TYPE

# 混合精度
# fp16 = dict(mode=AMP_TYPE.TORCH)

# 梯度积累
# gradient_accumulation = 4

# 梯度裁剪
# clip_grad_norm = 1.0

# 流水线并行
# Tensor并行
