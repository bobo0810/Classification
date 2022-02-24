## 亮点
- 抽象：分离业务与框架。
  - 扩展性强：各模块均统一入口。
  - 简单：新任务仅修改`Config/`即可。
- 模型：集成[Timm预训练模型库](https://github.com/rwightman/pytorch-image-models) ![Github stars](https://img.shields.io/github/stars/rwightman/pytorch-image-models.svg)
- 可视化: 集成[TensorBoard](https://github.com/tensorflow/tensorboard)![Github stars](https://img.shields.io/github/stars/tensorflow/tensorboard.svg)
<div align=center><img src="./Docs/source/imgs/tsdb.gif" ></div>

## 安装依赖

```bash
pip install    ./Package/*zip
pip install -r ./Package/requirements.txt 
```

## API文档

https://bclassification.readthedocs.io/

- 最佳实践
- 自定义数据集

- 自定义主干网络
- 自定义 图像增广 | 损失函数 | 优化器

  - 各自对应的`__init__.py`入口修改即可。

## 框架

```bash
├── Config
│   └── *.yaml 参数配置
│   └── *.txt  数据集列表 
├── DataSets
│   └── preprocess.py 图像增广入口
├── Models
│   ├── Backbone/__init__.py  主干网络入口
│   ├── Head/__init__.py      损失函数入口
│   ├── Optimizer/__init__.py 优化器入口
├── test.py
└── train.py
```

