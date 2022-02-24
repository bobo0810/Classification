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

## 示例(猫狗2分类)

#### 1. 训练

```python
python train.py  --yaml ./Config/train.yaml  --txt ./Config/train.txt
```

#### 2. 测试

```python
# yaml配置模型权重
python test.py   --yaml ./Config/test.yaml   --txt ./Config/test.txt
```

## API文档

https://bclassification.readthedocs.io/



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

- 自定义数据集

  （1）准备数据：格式类似`CatDog/`文件夹，并修改*.yaml的`prefix`属性。

  （2）划分数据集：`ExtraTools/split_imgs.py`划分训练集/测试集，生成*.txt。

- 自定义主干网络
  - 自定义：（1）仿照`Backbone/__init__.py`内`MyNet` （2）修改*.yaml内`backbone`属性。
  - timm库: （1）查询支持的[模型名称](https://rwightman.github.io/pytorch-image-models/)  （2）修改*.yaml内`backbone`属性。

- 自定义 图像增广 | 损失函数 | 优化器

  - 均在各自入口修改即可，即`__init__.py`。

