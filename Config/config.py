class Config(object):
    # ======数据集============
    size= [224,224] # 图像尺寸
    sampler="batch_balance" # 采样策略 
    txt="./Config/dataset.txt" # 数据集路径

    # ======模型============
    optimizer="sgd" # 优化器

    # 常规分类
    backbone="mynet" # 主干网络 
    loss="cross_entropy"  # 损失函数

    # 度量学习
    # backbone="mynet_metric" # 主干网络  
    # loss="arcface"  # 损失函数 

    # ======训练============
    lr=0.01
    batch=64
    epochs=80 
    scheduler="cosine"  # 学习率调度器

cfg=Config()