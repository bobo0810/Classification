# 图像识别框架Classification

[![Actions Status](https://github.com/bobo0810/Classification/workflows/build/badge.svg)](https://github.com/bobo0810/Classification/actions)

- 收录到[PytorchNetHub](https://github.com/bobo0810/PytorchNetHub)
- [更新日志](https://github.com/bobo0810/Classification/releases)
- 0.5.0版本开始，仅支持分布式训练。


## 亮点

|          | 功能                                                         | 备注                                                         |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 模块化    | 解耦业务与框架                                               | 1. 易用：新任务仅修改`Config/`即可。 <br/>2. 扩展性强：各模块均设统一入口。 |
| 模型     | 集成[Timm](https://github.com/rwightman/pytorch-image-models) | 1. 持续更新SOTA的预训练模型(600+)。<br/>2. 轻松定制模型。                                   |
| 训练 | 集成[ColossalAI](https://github.com/hpcaitech/ColossalAI) | 1. 大规模分布式并行、自动混合精度。<br/>2. 梯度积累、梯度裁剪等。 |
| 可视化   | 集成[TensorBoard](https://github.com/tensorflow/tensorboard) | 可视化参数、损失、图像、模型结构等。 |
| 部署 | 服务器/移动端加速                                                        | <img src="Docs/imgs/deploy.svg" style="zoom:50%;" /> |


## 支持任务

- 图像分类✅
- 度量学习/特征对比✅

## [Wiki文档](https://github.com/bobo0810/Classification/wiki)

- [最佳实践](https://github.com/bobo0810/Classification/wiki/%E6%9C%80%E4%BD%B3%E5%AE%9E%E8%B7%B5) ⭐️ ⭐️ ⭐️

- 自定义数据集

- 自定义主干网络

  (1)Timm库   (2)自定义   (3)基于Timm库定制

- 模型部署

  全流程支持 转换->加载->推理->验证误差。

## 可视化

<center>训练</center>

  <div align=center><img src="./Docs/imgs/tsdb.gif" width="500px"  height="500px"></div>

<center>测试:支持120+评价指标</center>

  <div align=center><img src="./Docs/imgs/matrix.jpg" width="1000px"  height="400px" ></div>

<center>预测:类激活图</center>

  <div align=center><img src="./Docs/imgs/cam_cat.jpg" ><img src="./Docs/imgs/cam_dog.jpg" ></div>

## 扩展框架

```bash
├── Config
│   └── config.py    训练配置
│   └── dataset.txt  数据集 
├── DataSets
│   └── preprocess.py 预处理入口
├── Models
│   ├── Backbone    主干网络入口
│   ├── Optimizer   优化器入口
│   ├── Loss        损失函数入口
│   ├── Backend     模型部署入口
│   ├── Scheduler   学习率调度器入口
```



## 训练配置

|   常规分类   | 属性  | 支持                                                         |
| ------------ | --------- | ------------------------------------------------------------ |
| 采样策略     | Sampler   | - normal     常规采样<br>- dataset_balance    类别平衡采样(数据集维度)  <br>- batch_balance    类别平衡采样(batch维度)⭐️        |
| 主干网络     | Backbone  | - [600+ SOTA预训练模型](https://github.com/bobo0810/Classification/wiki/%E8%87%AA%E5%AE%9A%E4%B9%89%E4%B8%BB%E5%B9%B2%E7%BD%91%E7%BB%9C)  |
| 损失函数     | Loss      | - cross_entropy<br>- label_smooth         |
| 学习率调度器      | Scheduler | - cosine⭐️ <br/>- multistep|
| 优化器       | Optimizer | - sgd<br/>- adam<br/>- lamb  |


| 度量学习 | 属性 | 支持                                                         |
| -------- | -------- | ------------------------------------------------------------ |
| 主干网络 | Backbone | - [600+ SOTA预训练模型](https://github.com/bobo0810/Classification/wiki/%E8%87%AA%E5%AE%9A%E4%B9%89%E4%B8%BB%E5%B9%B2%E7%BD%91%E7%BB%9C) |
| 损失函数 | Loss     | - cosface<br/>- arcface⭐️<br/>- subcenter_arcface<br/>- circleloss |

## 感谢

- 教程
  - [Timm快速上手](https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055)
  - [TensorRT安装教程](https://www.codeleading.com/article/48816068405/)
- 开源库
  - [Yolov5目标检测库](https://github.com/ultralytics/yolov5)![Github stars](https://img.shields.io/github/stars/ultralytics/yolov5.svg)
  - [Timm预训练模型库](https://github.com/rwightman/pytorch-image-models)![Github stars](https://img.shields.io/github/stars/rwightman/pytorch-image-models.svg)
  - [PyCM多类指标统计库](https://github.com/sepandhaghighi/pycm)![Github stars](https://img.shields.io/github/stars/sepandhaghighi/pycm.svg)
  - [torchinfo模型统计库](https://github.com/TylerYep/torchinfo)![Github stars](https://img.shields.io/github/stars/TylerYep/torchinfo.svg)
  - [pytorch-grad-cam类激活映射库](https://github.com/jacobgil/pytorch-grad-cam)![Github stars](https://img.shields.io/github/stars/jacobgil/pytorch-grad-cam.svg)
  - [pytorch-metric-learning度量学习库](https://github.com/KevinMusgrave/pytorch-metric-learning)![Github stars](https://img.shields.io/github/stars/KevinMusgrave/pytorch-metric-learning.svg)
  - [ColossalAI大规模分布式训练库](https://github.com/hpcaitech/ColossalAI)![Github stars](https://img.shields.io/github/stars/hpcaitech/ColossalAI.svg)
  

