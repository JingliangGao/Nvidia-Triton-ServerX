# Triton 推理服务
Triton推理服务是基于[NVIDIA Triton Inference Server](https://github.com/triton-inference-server)开发，将triton各个模块重新组织在一个工程中便于管理和开发。

- server[main commit 9ec820]:服务端
- client[main commit cb9ba08]:客户端
- core[main commit 13b6046]:核心库接口
- backend[main commit 30fa78a]:后端库
- common[main commit 578491f]:通用库
- third_party[main commit 3d456e0]:第三方依赖库
- framework_backends:推理框架

## 编译

### server

```
cmake -DCMAKE_INSTALL_PREFIX=`pwd`/install  -DTRITON_ENABLE_SERVER=ON -DTRITON_ENABLE_CLIENT=OFF  -DTRITON_ENABLE_ONNX_BACKEND=OFF ..
```

### client

```
cmake -DCMAKE_INSTALL_PREFIX=`pwd`/install -DTRITON_ENABLE_SERVER=OFF -DTRITON_ENABLE_CLIENT=ON -DTRITON_ENABLE_ONNX_BACKEND=OFF -DTRITON_ENABLE_CC_HTTP=ON -DTRITON_ENABLE_CC_GRPC=ON -DTRITON_ENABLE_EXAMPLES=ON -DTRITON_ENABLE_TESTS=ON -DTRITON_ENABLE_PYTHON_HTTP=ON -DTRITON_ENABLE_PYTHON_GRPC=ON  ..
```

### onnxruntime_backend

```
cmake -DCMAKE_INSTALL_PREFIX=`pwd`/install  -DTRITON_ENABLE_SERVER=OFF -DTRITON_ENABLE_CLIENT=OFF  -DTRITON_ENABLE_ONNX_BACKEND=ON  -DTRITON_ONNXRUNTIME_INCLUDE_PATHS=/home/kylin/bzm/project/onnxruntime/include/onnxruntime/core/session -DTRITON_ONNXRUNTIME_LIB_PATHS=/home/kylin/bzm/project/onnxruntime/build/Linux/RelWithDebInfo ..
```

### llamacpp_backend

```
cmake -DCMAKE_INSTALL_PREFIX=`pwd`/install  -DTRITON_ENABLE_SERVER=OFF -DTRITON_ENABLE_CLIENT=OFF  -DTRITON_ENABLE_ONNX_BACKEND=OFF -DTRITON_ENABLE_LLAMACPP_BACKEND=ON -DTRITON_LLAMACPP_INCLUDE_PATHS=/home/kylin/bzm/triton/llamacpp_backend/third_party/include -DTRITON_LLAMACPP_LIB_PATHS=/home/kylin/bzm/triton/llamacpp_backend/third_party/lib   ..
```

### python_backend

```
cmake -DCMAKE_INSTALL_PREFIX=`pwd`/install  -DTRITON_ENABLE_SERVER=OFF -DTRITON_ENABLE_CLIENT=OFF -DTRITON_ENABLE_GPU=OFF -DTRITON_ENABLE_PYTHON_BACKEND=ON  ..
```

### 测试
```
cmake -DCMAKE_INSTALL_PREFIX=`pwd`/install -DTRITON_ENABLE_SERVER=OFF -DTRITON_ENABLE_CLIENT=ON -DTRITON_ENABLE_ONNX_BACKEND=OFF -DTRITON_ENABLE_CC_HTTP=ON -DTRITON_ENABLE_CC_GRPC=ON -DTRITON_ENABLE_EXAMPLES=OFF -DTRITON_ENABLE_TESTS=ON -DTRITON_ENABLE_PYTHON_HTTP=ON -DTRITON_ENABLE_PYTHON_GRPC=ON -DTRITON_ENABLE_GPU=OFF -DTRITON_ENABLE_METRICS_GPU=OFF ..
```

## 使用