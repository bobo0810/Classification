# 自定义数据集

## 1. 准备数据

假设数据集根路径为`/home/xxx/CatDog`  ，格式如下

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

2. 划分数据集，默认`Config/`下生成train.txt、test.txt。

```bash
python  ./ExtraTools/split_imgs.py  --ImgsPath=/home/xxx/CatDog  --Ratio=[0.8,0.2]  --Verify
```

- --ImgsPath    数据集根路径
- --Ratio           各类别均按指定比例分配train:test，默认[0.8, 0.2]
- --Verify          验证图像完整性(耗时，可选)



## 二. 配置

修改`Config/train.yaml`

```yaml
DataSet:
  prefix: /home/xxx/CatDog # 数据集根路径 
  category: {"cat":0,"dog":1} # 类别
```

