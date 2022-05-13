# ======数据集============
Size= [224,224] # 图像尺寸
Sampler="batch_balance" # 采样策略 
Txt="./Config/dataset.txt" # 数据集路径

# ======模型============
Optimizer="sgd" # 优化器

# 常规分类                  # 度量学习
Backbone="mynet"           # Backbone="mynet_metric"    # 主干网络
Loss="cross_entropy"       # Loss="arcface"             # 损失函数 

# ======训练============
LR=0.01     # 学习率
Batch=64    # 批次
Epochs=80   # 总轮数


# ======分布式============
from colossalai.amp import AMP_TYPE
fp16 = dict(mode=AMP_TYPE.TORCH) # 开启混合精度
