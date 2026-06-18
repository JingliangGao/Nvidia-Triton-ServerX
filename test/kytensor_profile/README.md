# Model Analyzer

Model Analyzer - 用于分析 Kytensor 模型性能的工具。

## 功能特性

- 模型加载时间测量
- 模型内存使用跟踪
- 推理延迟和吞吐量测量
- 峰值内存使用监控
- 模型卸载时间跟踪
- 服务器 CPU 和系统资源监控
- 支持多种模型类型:
  - **OCR 模型**（如 `ocr_ppocr`）
  - **GTE 嵌入模型**（如 `embd_gte-base_uint8-text`）
  - **CN-CLIP 文本模型**（如 `embd_cn-clip_512-uint8-text`）
  - **CN-CLIP 图像模型**（如 `embd_cn-clip_512-uint8-image`）
- 自动数据预处理（图像和文本）
- JSON 格式输出指标和输入数据

## 安装

### 从源码安装

```bash
pip install .
```

### 开发模式安装

```bash
pip install -e .
```

## 使用方法

安装后，可以使用以下命令运行分析器：

```bash
model-analyzer --model-repository /path/to/repo --triton-endpoint localhost:8001
```

### 命令行选项

| 选项 | 描述 | 默认值 |
|--------|-------------|---------|
| `--model-repository` | Triton 模型仓库路径 | `/usr/share/kylin-ai/model-repository` |
| `--triton-endpoint` | Triton 服务器地址 | `localhost:8001` |
| `--ocr-image-dir` | OCR 模型原始图像数据目录 | `test_data/ocr_image/` |
| `--gte-text-dir` | GTE 模型原始文本数据目录 | `test_data/gte_text/` |
| `--cn-clip-text-dir` | CN-CLIP Text 模型原始文本数据目录 | `test_data/cn_clip_text/` |
| `--cn-clip-image-dir` | CN-CLIP Image 模型原始图像数据目录 | `test_data/cn_clip_image/` |
| `--skip-ocr` | 跳过 OCR 模型测试 | - |
| `--skip-gte` | 跳过 GTE 模型测试 | - |
| `--skip-cn-clip-text` | 跳过 CN-CLIP Text 模型测试 | - |
| `--skip-cn-clip-image` | 跳过 CN-CLIP Image 模型测试 | - |

### 跳过特定模型

可以使用 `--skip-*` 标志跳过特定模型的测试：

```bash
# 跳过 OCR 模型，仅测试 GTE 和 CN-CLIP
model-analyzer --skip-ocr

# 跳过除 OCR 之外的所有模型
model-analyzer --skip-gte --skip-cn-clip-text --skip-cn-clip-image
```

## 输出指标

分析器以 JSON 格式输出以下指标：

### 模型信息

| 指标 | 描述 |
|--------|-------------|
| `model_name` | 模型名称 |
| `model_repository` | 模型仓库绝对路径 |
| `protocol` | 通信协议 |
| `endpoint` | 服务器端点 |
| `batch_size` | 批处理大小 |
| `num_requests` | 推理请求数量 |

### 模型加载性能

| 指标 | 描述 |
|--------|-------------|
| `model_load_time_s` | 模型加载时间（秒） |
| `model_load_memory_mb` | 模型加载期间使用的内存（MB） |

### 模型卸载性能

| 指标 | 描述 |
|--------|-------------|
| `model_unload_time_s` | 模型卸载时间（秒） |

### 推理性能

| 指标 | 描述 |
|--------|-------------|
| `peak_inference_memory_mb` | 推理期间峰值内存（MB） |
| `inference_total_time_s` | 总推理时间（秒） |
| `inference_average_latency_ms` | 平均推理延迟（毫秒） |
| `inference_throughput_inferences_per_s` | 推理吞吐量（推理次数/秒） |

### 服务器资源使用

| 指标 | 描述 |
|--------|-------------|
| `server_cpu_uss_mb` | 服务器进程内存使用量（MB） |
| `server_available_ram_mb` | 系统可用内存（MB） |
| `server_cpu_percent` | 服务器进程 CPU 使用率（%） |
| `system_cpu_percent` | 系统 CPU 使用率（%） |

## 数据处理器

分析器包含针对不同模型类型的内置数据处理器：

| 处理器 | 模型类型 | 输入 | 描述 |
|-----------|-------------|-------|-------------|
| `OCRDataProcessor` | `ocr`, `ocr_ppocr` | 图像（PNG、JPG、JPEG、BMP、TIFF、WebP） | 将图像转换为 BGR 格式 |
| `GTETextDataProcessor` | `gte`, `embd_gte-base_uint8-text` | 文本（.txt） | 使用 HuggingFace tokenizer 或简单字符级 tokenizer 对文本进行分词 |
| `CNCLIPTextDataProcessor` | `cn-clip`, `embd_cn-clip_512-uint8-text` | 文本（.txt） | 为 CN-CLIP 文本模型对文本进行分词 |
| `CNCLIPImageDataProcessor` | `cn-clip-image`, `embd_cn-clip_512-uint8-image` | 图像（PNG、JPG、JPEG、BMP、TIFF、WebP） | 将图像调整大小并归一化为 NCHW 格式 |

### 自定义数据处理器

可以使用以下方法注册自定义数据处理器：

```python
from model_analyzer.process_data import DataProcessorFactory, DataProcessor

class MyProcessor(DataProcessor):
    def preprocess(self, input_path):
        # 你的预处理逻辑
        pass

DataProcessorFactory.register_processor("my_model", MyProcessor)
```

## 依赖

- numpy >= 1.20.0
- psutil >= 5.8.0
- tritonclient[all] >= 2.20.0
- Pillow >= 8.0.0
- transformers >= 4.0.0