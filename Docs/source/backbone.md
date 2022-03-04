# 自定义主干网络

匹配优先级：自定义 > Timm库

## 方案1：Timm库（推荐⭐️）

持续更新SOTA的预训练模型(当前模型数>600)。[官方文档](https://rwightman.github.io/pytorch-image-models/) 

**适用场景**：仅更改预测类别数。


1. 挑选SOTA模型

   ```python
   import timm
   from pprint import pprint
   
   model_names = timm.list_models(pretrained=True) # 查询模型名称
   # model_names = timm.list_models('*resne*t*') # 支持通配符
   pprint(model_names)
   >>> ['resnet18','resnet50',...]
   ```

2. 修改`Config/train.yaml`

   ```yaml
   Models: 
     backbone: resnet18 
   ```



## 方案2：自定义

**适用场景**：自定义结构 或 基于已有结构定制。

1. 定义网络结构

   参考`/Models/Backbone/mynet.py`

2. 修改`Config/train.yaml`

   ```yaml
   Models: 
     backbone: mynet
   ```




## 附：基于Timm库定制

[官方文档](https://fastai.github.io/timmdocs/create_model#Turn-any-model-into-a-feature-extractor)

基本结构

- 输入：3x224x224的图像

- 卷积层：降采样2、4、8、16、32倍，输出channel x 7 x 7特征。

- 全局池化层：全局池化，输出N维特征(N=通道数)。
- 分类层：输出最终类别维度。

#### 特征提取器

```python
# 创建 无全局池化和分类层的网络, 将输出卷积层最后特征 [batch, channel, 7, 7]
features = timm.create_model("resnet18", pretrained=True, num_classes=0, global_pool='') 

# 创建 无分类层的网络, 将输出全局池化后的特征 [batch, channel]
features = timm.create_model("resnet18", pretrained=True, num_classes=0) 
```

#### 获取中间层特征

```python
# 将输出 卷积层5次降采样的特征
features = timm.create_model("resnet18", pretrained=True, features_only=True)

img = torch.ones((1, 3, 224, 224))
features(img)  # list
# [batch,x,112,112]
# [batch,x,56,56] 
# [batch,x,28,28]  
# [batch,x,14,14] 
# [batch,x,7,7] 
```

