## 亮点

|          | 功能                                                         | 备注                                                         |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 抽象     | 解耦业务与框架                                               | 1. 易用：新任务仅修改`Config/`即可。 <br/>2. 扩展性强：各模块均设统一入口。 |
| 模型     | 集成[Timm预训练模型库](https://github.com/rwightman/pytorch-image-models) ![Github stars](https://img.shields.io/github/stars/rwightman/pytorch-image-models.svg) | 持续更新SOTA的预训练模型。（模型数>600）                                   |
| 可视化   | 集成[TensorBoard](https://github.com/tensorflow/tensorboard)![Github stars](https://img.shields.io/github/stars/tensorflow/tensorboard.svg) | 可视化参数、损失、训练图像、模型结构等。                     |
| 部署 | 服务器/移动端加速                                                        | 1. ONNX ✅<br/>2. TensorRT✅<br/>3. todo |

<div align=center><img src="./Docs/source/imgs/tsdb.gif" ></div>




## 快速开始
1. 下载源码，安装依赖。
    ```bash
    pip install    ./Package/*zip
    pip install -r ./Package/requirements.txt 
    ```
    注：示例数据集和参数已配好

2. 训练：执行`python train.py`
3. 测试：`Config/test.yaml`配置权重，执行`python test.py`

## API文档

https://bclassification.readthedocs.io/ 

文档内容如下：

- 最佳实践(训练+测试)

- 自定义数据集

- 自定义主干网络
  - 方案1：Timm库
  - 方案2：自定义
  - 附：基于Timm库定制

- 自定义 图像增广 | 损失函数 | 优化器| 模型转换
  - 在各自入口修改即可。
- 模型部署
  - Torch转为ONNX并推理
  - ONNX转为TensorRT并推理

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
│   ├── Backend               模型部署入口
├── export.py
├── test.py
└── train.py
```



## 参考
- [pytorch-image-models](https://github.com/rwightman/pytorch-image-models)

- [yolov5](https://github.com/ultralytics/yolov5)

- [Getting Started with PyTorch Image Models (timm): A Practitioner’s Guide](https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055)

- [TensorRT8安装教程](https://www.codeleading.com/article/48816068405/)


