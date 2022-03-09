# 自定义数据集

## 一. 准备数据

1.假设数据集根路径为`/home/xxx/CatDog/`  ，格式如下

```bash
├── cat
    ├── aaa.jpg
    ├── bbb.jpg
    ├── ....
├── dog
    ├── ccc.jpg
    ├── bbb.jpg
    ├── ....
```

2.划分数据集，默认`Config/`下生成train.txt、test.txt、labels.txt。

```bash
python  ./Utils/split_imgs.py  --ImgsPath=/home/xxx/CatDog/ 
```

- ImgsPath    数据集根路径
- Ratio       各类别均按指定比例分配train:test，默认0.8
- Verify      验证图像完整性(耗时，可选)
- TxtPath     train/test/labels.txt保存路径,默认保存到`Config/`



## 二. 配置

修改`Config/train.yaml`

```yaml
DataSet:
  prefix: /home/xxx/CatDog/ # 数据集根路径 
  size: [224,224]  # 训练尺寸
```

