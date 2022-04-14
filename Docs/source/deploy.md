# 模型部署

推荐`export.py`查看完整参数

## 1.TorchScript
```bash
python export.py  --weights="xxx.pt"  --torch2script
```
- torch2script       (可选)转为torchscript
- script_gpu         (可选)导出GPU模型，默认CPU模型
```bash
****************************
TorchScript export success, saved as /xxx/mynet.torchscript

****************************
output_torch - output_script =  0.0
```

## 2. ONNX

```bash
python export.py  --weights="xxx.pt" --torch2onnx
```
- torch2onnx   (可选)转为onnx
- simplify     (可选)简化onnx模型
- dynamic      (可选)batch轴设为动态



控制台输出

```bash
****************************
ONNX export success, saved as /xxx/mynet.onnx
Visualize onnx with https://github.com/lutzroeder/netron.

****************************
output_torch - output_onnx =  5.379319e-06
```



## 3. TensorRT

注意：（1）传入的onnx模型必须固定尺度 （2）TensorRT版本>=8.0

```bash
python export.py --weights="xxx.pt" --torch2onnx --onnx2trt 
```
- onnx2trt  （可选）onnx是否转为tensorrt
- fp16       (可选）开启fp16预测



控制台输出

```bash
****************************
ONNX export success, saved as /xxx/mynet.onnx
Visualize onnx with https://github.com/lutzroeder/netron.

****************************
TensorRT export success, saved as /xxx/mynet.trt

****************************
output_torch - output_onnx =  5.379319e-06
output_torch - output_trt =  8.6221844e-05
```

