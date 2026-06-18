# Triton LlamaCpp Backend

基于llama.cpp实现的Triton Inference Server后端，提供大模型推理服务，暂时完成以下功能:

* 支持同步异步推理、流式推理
* 实现completions、tokenize、chat功能
* 支持多用户并发
* 支持continuous batching

## 使用

### 依赖安装

不用编译省略这一步

```
sudo apt-get install openssl libssl-dev rapidjson-dev libnuma-dev libre2-dev libb64-dev libopencv-dev
```

```
pip install grpcio-tools
```

### 后端部署

* onnxruntim

```
cp libtriton_onnxruntime.so /opt/tritonserver/backends/onnxruntime/
```

* llamacpp

在模型仓库路径下创建以下llamacpp目录，将libtriton_llamacpp.so放入1目录下，config.pbtxt放入llamacpp目录下。

llamacpp
├── 1
│   └── libtriton_llamacpp.so
└── config.pbtxt

### 服务端部署

运行tritonserver

```
//先拷贝libtritonserver库
cp libtritonserver.so /lib/x86_64-linux-gnu/
//再运行
./tritonserver --model-repository PATH/model_repository/
```

### 客户端部署

Triton客户端支持C++、python、java三种语言，使用方法都类似，以下都以python为例说明用法。

#### python客户端安装

```
pip install tritonclient-2.49.0-py3-none-manylinux1_x86_64.whl
```

#### 例子说明

* llamacpp_client

该例子展示了no streaming用法，提供了同步和异步推理。

* llamacpp_client_stream

该例子展示了streaming推理用法。

* llamacpp_client_chat

该例子展示了对话功能用法。

* cc-clients

triton client编译出的c++库和例子

* python-clients

triton client编译出的python库和例子

## 测试

### 推理性能测试

* 硬件
  CPU:Intel E5-2680
  GPU:NVIDIA RTX A5000
* 系统
  ubuntu22.04

|       服务       | 推理模式 | 模型                         | 提示词 | prompt(tokens/s) | generation(tokens/s) |
| :--------------: | -------- | ---------------------------- | ------ | :--------------: | :------------------: |
| llama.cpp server | CPU      | qwen1.5_7b-model-Q4_K_M.gguf | 你是谁 |      31.55      |         3.97         |
| triton llamacpp | CPU      | qwen1.5_7b-model-Q4_K_M.gguf | 你是谁 |      31.13      |         3.96         |
| llama.cpp server | GPU      | qwen1.5_7b-model-Q4_K_M.gguf | 你是谁 |      72.72      |         2.19         |
| triton llamacpp | GPU      | qwen1.5_7b-model-Q4_K_M.gguf | 你是谁 |      503.06      |        63.21        |

### 服务性能测试

暂未测试

