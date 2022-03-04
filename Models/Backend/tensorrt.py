import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import numpy as np


class TensorrtBackend:
    """
    TensorRT后端
    """

    def __init__(self):
        pass

    @staticmethod
    def convert(onnx_weights, trt_weights, fp16=False):
        """
        onnx模型转为tensorrt模型
        注意：
        1. 仅适用TensorRT V8以上版本
        2. 仅支持固定输入尺度

        onnx_weights: onnx模型路径
        trt_weights: trt引擎文件的保存路径
        fp16: 是否开启半精度预测
        """

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(EXPLICIT_BATCH)
        parser = trt.OnnxParser(network, TRT_LOGGER)

        config = builder.create_builder_config()
        config.max_workspace_size = 1 * (1 << 30)  # 1 * GiB
        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        with open(onnx_weights, "rb") as model:
            assert parser.parse(model.read())
            serialized_engine = builder.build_serialized_network(network, config)

        with open(trt_weights, "wb") as f:
            f.write(serialized_engine)  # 序列化保存为trt引擎文件
        print("*" * 28)
        print("TensorRT export success, saved as %s" % trt_weights)

    @staticmethod
    def infer(weights, imgs, output_shape):
        """
        加载trt引擎并推理

        weights(str): tensorrt权重路径
        img(numpy): [B,C,H,W]
        output_shape : 期望输出的形状
        """
        model = TrtModel(weights)  # 初始化trt引擎
        output = model(imgs)  # 推理
        model.destroy()  # 释放资源
        return output.reshape(output_shape)  # trt输出一维数组，再reshape为期望形状


class TrtModel:
    def __init__(self, weights):
        """初始化trt模型"""
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(weights, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            print("bingding:", binding, engine.get_binding_shape(binding))
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size

    def __call__(self, img_np_nchw):
        """
        TensorRT推理

        img_np_nchw: 输入图像
        """
        self.ctx.push()

        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings

        np.copyto(host_inputs[0], img_np_nchw.ravel())
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        context.execute_async(
            batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle
        )
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        stream.synchronize()
        self.ctx.pop()
        return host_outputs[0]

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
