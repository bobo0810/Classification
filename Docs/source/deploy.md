# 模型部署

## 1. Torch->ONNX

```bash
python export.py  --weights="xxx.pth" --simplify
```

- img_size           推理尺寸，默认[1, 3, 224, 224]
- weights           （必选）模型权重，已包含网络结构和参数，可直接加载。

- onnx相关参数
  - simplify     (可选)简化onnx模型，默认关闭
  - dynamic   (可选)batch轴设为动态，默认关闭



控制台输出

```bash
****************************
ONNX export success, saved as /xxx/mynet_099.onnx
Visualize onnx with https://github.com/lutzroeder/netron.

****************************
output_torch - output_onnx =  5.379319e-06
```



## 2. ONNX-> TensorRT

注意：（1）传入的onnx模型必须固定尺度 （2）TensorRT版本>=8.0

```bash
python export.py --weights="xxx.pth" --simplify --onnx2trt 
```

- tensorrt相关参数
  - onnx2trt  （可选）onnx是否转为tensorrt，默认关闭
  - fp16       (可选）开启fp16预测，默认关闭



控制台输出

```bash
****************************
ONNX export success, saved as /xxx/mynet_099.onnx
Visualize onnx with https://github.com/lutzroeder/netron.

****************************
TensorRT export success, saved as /xxx/mynet_099.trt

****************************
output_torch - output_onnx =  5.379319e-06
output_torch - output_trt =  8.6221844e-05
```

