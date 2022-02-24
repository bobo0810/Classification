# 自定义主干网络

匹配优先级：自定义 > Timm库

## 方案1：自定义

>  假设模型文件为mynet.py 、模型名称为MyNet

1. 定义模型

   （1）`/Backbone/`新建mynet.py

   （2）`/Backbone/__init__.py`的init_model()加载MyNet

2. 修改`Config/train.yaml`

   ```yaml
   Models: 
     backbone: MyNet
   ```

   

## 方案2：Timm库

持续提供最新SOTA的预训练模型。[官方文档](https://rwightman.github.io/pytorch-image-models/)

1. 查询网络名称 

   ```python
   import timm
   from pprint import pprint
   
   # 所有预训练模型
   model_names = timm.list_models(pretrained=True)
   pprint(model_names)
   >>> ['resnet18',
    'resnet50',
   ...
   ]
   
   # 通配符查询
   model_names = timm.list_models('*resne*t*')
   pprint(model_names)
   >>> ['cspresnet50',
    'cspresnet50d',
   ...
   ]
   ```

2. 修改`Config/train.yaml`

   ```yaml
   Models: 
     backbone: resnet18 
   ```

## 方案3：TorchHub库

[地址](https://pytorch.org/hub/research-models)

