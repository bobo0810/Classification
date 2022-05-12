# ======数据集============
Size= [224,224] # 图像尺寸
Sampler="batch_balance" # 采样策略 
Txt="./Config/dataset.txt" # 数据集路径

# ======模型============
Optimizer="sgd" # 优化器

# 常规分类
Backbone="mynet" # 主干网络 
Loss="cross_entropy"  # 损失函数

# 度量学习
# Backbone="mynet_metric" # 主干网络  
# Loss="arcface"  # 损失函数 

# ======训练============
LR=0.01     # 学习率
Batch=64    # 批次
Epochs=80   # 总轮数
Scheduler="cosine"  # 学习率调度器
