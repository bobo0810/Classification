
## 安装

```bash
pip install -r ./Package/requirements.txt 
```

# 训练

### 数据

pass

### 训练

pass

### 测试

pass

## 框架

```bash
├── Config
│   └── *.yaml 配置文件
│   └── *.txt  图像列表 
├── DataSets
│   └── transforms.py 图像增广入口
├── Models
│   ├── Backbone/__init__.py  主干网络入口
│   ├── Head/__init__.py      损失函数入口
│   ├── Optimizer/__init__.py 优化器入口
├── Utils
├── test.py
└── train.py
```

- 模型：集成[Timm预训练库](https://github.com/rwightman/pytorch-image-models) ![Github stars](https://img.shields.io/github/stars/rwightman/pytorch-image-models.svg)
- 可视化: 集成[TensorBoard](https://github.com/tensorflow/tensorboard)![Github stars](https://img.shields.io/github/stars/tensorflow/tensorboard.svg)

- 数据增广：TODO