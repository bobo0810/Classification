# 更多模型转换参数请访问: https://www.yuque.com/mnn/cn/model_convert

import MNN
import numpy as np
import MNN
import os


class MNNBackbend:
    """
    MNN后端

    代码参考 官方文档https://www.yuque.com/mnn/cn/vg3to5
    """

    def __init__(self):
        pass

    @staticmethod
    def convert(onnx_weights, weights, fp16=False):
        """
        onnx模型转为MNN模型

        onnx_weights: onnx模型路径
        weights: MNN文件的保存路径
        fp16:  将float32参数保存为float16,模型将减小一半,精度基本无损

        注: mnn python版转换不支持权值量化参数weightQuantBits
        """
        shell = "mnnconvert -f ONNX --modelFile %s --MNNModel %s  --bizCode biz  " % (
            onnx_weights,
            weights,
        )
        shell += "--fp16" if fp16 else ""
        os.system(shell)
        print("*" * 28)
        print("MNN export success, saved as %s" % weights)

    @staticmethod
    def infer(weights, imgs, output_shape):
        """
        加载mnn并推理

        weights(str): mnn权重路径
        img(numpy): [B,C,H,W]
        output_shape : 期望输出的形状
        """
        # ---------------------------初始化模型-------------------------------------
        interpreter = MNN.Interpreter(weights)  # 加载模型（解释器）
        interpreter.setCacheFile(".tempcache")  # 设置缓存，提升初始化速度
        config = {}
        config["precision"] = "low"  # 精度
        session = interpreter.createSession()  # 创建会话
        input_tensor = interpreter.getSessionInput(session)  # 定义模型输入
        # -------------------------------图像预处理---------------------------------
        img = imgs.astype(np.float32)
        # Caffe格式NCHW
        B, C, H, W = imgs.shape
        tmp_input = MNN.Tensor(
            (B, C, H, W), MNN.Halide_Type_Float, img, MNN.Tensor_DimensionType_Caffe
        )
        input_tensor.copyFrom(tmp_input)  # 拷贝给模型输入
        # -------------------------------推理---------------------------------
        interpreter.runSession(session)  # 推理
        output_tensor = interpreter.getSessionOutput(session)  # 得到输出

        # 构建临时tensor,保存输出结果 NCHW
        tmp_output = MNN.Tensor(
            output_shape,
            MNN.Halide_Type_Float,
            np.ones(output_shape).astype(np.float32),
            MNN.Tensor_DimensionType_Caffe,
        )
        output_tensor.copyToHostTensor(tmp_output)
        result = np.array(tmp_output.getData(), dtype=float)
        return result
