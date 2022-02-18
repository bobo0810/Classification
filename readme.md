
## 安装

```bash
pip install -r ./Package/requirements.txt 
```

## 划分数据集
```bash
python ./ExtraTools/split_imgs.py --ImgsPath="/HOME/IMAGE_PATH"  --Ratio=[0.8,0.2]  --Verify
```

- ImgsPath 数据集根路径 
- Ratio  train与test的划分比例 
- Verify(可选) 验证图像完整性(耗时)

# 总结
- 数据增广：集成Augment增广库
- 模型：集成Timm预训练库
- 可视化: 集成TensorBoard

# 准备数据，开始训练