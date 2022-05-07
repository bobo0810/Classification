# 最佳实践

## 一. 准备数据

> 根目录已包含`CatDog/`数据集，且`Config/`已生成`train.txt`、`test.txt`、`labels.txt`。

## 二. 训练

1. `Config/train.yaml`参数已配好
    
2. 开始训练

   ```bash
   python train.py  --yaml ./Config/train.yaml  --txt ./Config/train.txt
   ```

   控制台输出

   ```bash
   ****************************
   TensorBoard | Checkpoint save to  /xxx/ExpLog/2022-02-24_15:20:41/ 
   
   ****************************
   The nums of trainSet: 144
   The nums of each class:  {'cat': 73, 'dog': 71} 
   
   ****************************
   The nums of valSet: 16
   The nums of each class:  {'dog': 11, 'cat': 5} 
   
   start epoch 0/50...
   ```

3. TensorBoard可视化



## 三. 测试

1. `Config/test.yaml`需配置模型权重

2. 测试

   ```bash
   python test.py   --yaml ./Config/test.yaml   --txt ./Config/test.txt
   ```

   控制台输出 

   ```bash
   checkpoint load /xxx/resnet18.pth
   ****************************
   The nums of testSet: 40
   The nums of each class:  {'dog': 20, 'cat': 20}  
   
   accuracy is 0.975
   
   Predict  cat        dog        
   Actual
   cat        20       0        
   
   dog        1        19       
   
   
   Predict    cat          dog          
   Actual
   cat          1.0        0.0        
   
   dog          0.05       0.95 
   ```

