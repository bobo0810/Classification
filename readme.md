## 亮点

|          | 功能                                                         | 备注                                                         |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 抽象     | 解耦业务与框架                                               | 1. 易用：新任务仅修改`Config/`即可。 <br/>2. 扩展性强：各模块均设统一入口。 |
| 模型     | 集成[Timm预训练模型库](https://github.com/rwightman/pytorch-image-models) ![Github stars](https://img.shields.io/github/stars/rwightman/pytorch-image-models.svg) | 持续更新SOTA的预训练模型。（模型数>600）                                   |
| 可视化   | 集成[TensorBoard](https://github.com/tensorflow/tensorboard)![Github stars](https://img.shields.io/github/stars/tensorflow/tensorboard.svg) | 可视化参数、损失、训练图像、模型结构等。                     |
| 部署 | 服务器/移动端加速                                                        | 1. Torch->ONNX ✅<br/>2. ONNX -> TensorRT<br/>3. ONNX -> TensorFlow pb<br/>4. ... |

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

  > 在各自对应的`__init__.py`入口修改即可。

## 框架

```bash
├── Config
│   └── *.yaml 训练参数
│   └── *.txt  数据集 
├── DataSets
│   └── preprocess.py 图像增广入口
├── Models
│   ├── Backbone/__init__.py  主干网络入口
│   ├── Optimizer/__init__.py 优化器入口
│   ├── Loss/__init__.py      损失函数入口
├── export.py
├── test.py
└── train.py
```

