
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
