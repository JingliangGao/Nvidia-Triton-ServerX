#!/usr/bin/env python3

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

try:
    import numpy as np
except ImportError as exc:
    raise RuntimeError(
        "运行此简化分析器需要 numpy。"
        "在使用 entrypoint.py 之前，请在你的 Python 环境中安装它。"
    ) from exc

try:
    from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException
    from model_analyzer.output.file_writer import FileWriter
    from model_analyzer.triton.client.client_factory import TritonClientFactory
    from model_analyzer.triton.server.server_config import TritonServerConfig
    from model_analyzer.triton.server.server_local import TritonServerLocal, find_kytensor_process
except ImportError:
    from model_analyzer_exceptions import TritonModelAnalyzerException
    from output.file_writer import FileWriter
    from triton.client.client_factory import TritonClientFactory
    from triton.server.server_config import TritonServerConfig
    from triton.server.server_local import TritonServerLocal, find_kytensor_process

LOGGER_NAME = "model_analyzer_logger"
logging.basicConfig(format="[Model Analyzer] %(message)s")
logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.INFO)

TRITON_DTYPE_TO_NUMPY = {
    "TYPE_BOOL": np.bool_,
    "TYPE_UINT8": np.uint8,
    "TYPE_UINT16": np.uint16,
    "TYPE_UINT32": np.uint32,
    "TYPE_UINT64": np.uint64,
    "TYPE_INT8": np.int8,
    "TYPE_INT16": np.int16,
    "TYPE_INT32": np.int32,
    "TYPE_INT64": np.int64,
    "TYPE_FP16": np.float16,
    "TYPE_FP32": np.float32,
    "TYPE_FP64": np.float64,
    "TYPE_STRING": np.object_,
}


def get_default_endpoint(protocol: str, endpoint: Optional[str]) -> str:
    """获取默认端点。"""
    if endpoint:
        return endpoint
    return "localhost:8001" if protocol == "grpc" else "localhost:8000"


def normalize_shape(
    shape: List[Any], batch_size: int, payload_shape: Optional[List[int]] = None
) -> List[int]:
    """
    标准化形状。

    Args:
        shape: 原始形状。
        batch_size: 批处理大小。
        payload_shape: 负载形状。

    Returns:
        标准化后的形状列表。
    """
    normalized = []
    if len(shape) == 0:
        if payload_shape:
            return [int(dim) for dim in payload_shape]
        return [batch_size]

    for idx, dim in enumerate(shape):
        if dim is None or int(dim) == -1 or int(dim) == 0:
            if payload_shape is not None and idx < len(payload_shape):
                normalized.append(int(payload_shape[idx]))
            elif idx == 0:
                normalized.append(batch_size)
            else:
                normalized.append(1)
        else:
            normalized.append(int(dim))
    return normalized


def load_input_payload(file_path: str) -> Dict[str, Any]:
    """加载输入负载。"""
    with open(file_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise TritonModelAnalyzerException("输入数据 JSON 必须是以输入名称为键的对象。")
    return payload


def load_input_payloads(input_path: str) -> List[Dict[str, Any]]:
    """从 JSON 文件或包含 JSON 文件的目录加载输入负载。"""
    payloads = []
    if os.path.isdir(input_path):
        json_files = sorted([f for f in os.listdir(input_path) if f.endswith(".json")])
        if not json_files:
            raise TritonModelAnalyzerException(f"目录中没有 JSON 文件: {input_path}")
        logger.info(f"在 {input_path} 中找到 {len(json_files)} 个 JSON 文件")
        for json_file in json_files:
            file_path = os.path.join(input_path, json_file)
            try:
                payload = load_input_payload(file_path)
                payloads.append(payload)
            except Exception as e:
                logger.warning(f"加载 {file_path} 失败: {e}")
    else:
        payload = load_input_payload(input_path)
        payloads.append(payload)
    return payloads


def model_config_dtype_to_numpy(dtype: str) -> type:
    """将 Triton 数据类型转换为 numpy 类型。"""
    if dtype not in TRITON_DTYPE_TO_NUMPY:
        raise TritonModelAnalyzerException(f"不支持的 Triton 数据类型: {dtype}")
    return TRITON_DTYPE_TO_NUMPY[dtype]


def triton_dtype_to_infer_dtype(dtype: str) -> str:
    """将 Triton 数据类型转换为推理数据类型。"""
    return dtype.replace("TYPE_", "") if dtype.startswith("TYPE_") else dtype


def build_input_tensor(input_config: Dict[str, Any], batch_size: int, payload: Optional[Dict[str, Any]]) -> tuple:
    """
    构建输入张量。

    Returns:
        包含 (name, shape, dtype, data) 的元组。
    """
    name = input_config["name"]
    payload_shape = None
    if payload and name in payload:
        payload_shape = list(np.asarray(payload[name]).shape)

    shape = normalize_shape(input_config.get("dims", []), batch_size, payload_shape)
    dtype = input_config.get("data_type", "TYPE_FP32")
    np_dtype = model_config_dtype_to_numpy(dtype)

    if payload and name in payload:
        data = np.asarray(payload[name], dtype=np_dtype)
        input_dims = input_config.get("dims", [])
        for idx, dim in enumerate(input_dims):
            if dim is None or int(dim) == -1 or int(dim) == 0:
                continue
            if idx >= len(data.shape) or data.shape[idx] != shape[idx]:
                raise TritonModelAnalyzerException(
                    f"输入 '{name}' 的形状 {list(data.shape)} 无效；期望形状 {shape}，在索引 {idx} 处固定维度为 {dim}。"
                )
        if list(data.shape) != shape:
            data = data.reshape(shape)
    else:
        if np_dtype == np.object_:
            data = np.full(shape, "", dtype=np_dtype)
        else:
            data = np.zeros(shape, dtype=np_dtype)

    return name, shape, triton_dtype_to_infer_dtype(dtype), data


def create_infer_inputs(model_config: Dict[str, Any], batch_size: int, payload: Optional[Dict[str, Any]], protocol: str) -> List[Any]:
    """创建推理输入对象。"""
    if protocol == "grpc":
        import tritonclient.grpc as grpcclient

        InferInput = grpcclient.InferInput
    else:
        import tritonclient.http as httpclient

        InferInput = httpclient.InferInput

    inputs = []
    for input_config in model_config.get("input", []):
        name, shape, dtype, data = build_input_tensor(input_config, batch_size, payload)
        infer_input = InferInput(name, shape, dtype)
        infer_input.set_data_from_numpy(data)
        inputs.append(infer_input)
    return inputs


def create_infer_outputs(model_config: Dict[str, Any], protocol: str) -> List[Any]:
    """为所有模型输出创建 InferRequestedOutput 对象。"""
    if protocol == "grpc":
        import tritonclient.grpc as grpcclient

        InferRequestedOutput = grpcclient.InferRequestedOutput
    else:
        import tritonclient.http as httpclient

        InferRequestedOutput = httpclient.InferRequestedOutput

    outputs = []
    for output_config in model_config.get("output", []):
        name = output_config["name"]
        infer_output = InferRequestedOutput(name)
        outputs.append(infer_output)
    return outputs


def create_server_config(model_repository: str, protocol: str, endpoint: str) -> TritonServerConfig:
    """创建服务器配置。"""
    server_config = TritonServerConfig()
    server_config["model-repository"] = os.path.abspath(model_repository)
    server_config["model-control-mode"] = "explicit"
    server_config["strict-model-config"] = "false"
    server_config["allow-metrics"] = "true"
    if protocol == "grpc":
        server_config["allow-grpc"] = "true"
        server_config["grpc-address"] = "0.0.0.0"
        server_config["grpc-port"] = endpoint.split(":")[-1]
    else:
        server_config["allow-http"] = "true"
        server_config["http-address"] = "0.0.0.0"
        server_config["http-port"] = endpoint.split(":")[-1]
    return server_config


def get_client(protocol: str, endpoint: str) -> Any:
    """获取客户端。"""
    if protocol == "grpc":
        return TritonClientFactory.create_grpc_client(server_url=endpoint)
    return TritonClientFactory.create_http_client(server_url=endpoint)


def write_output(output_path: Optional[str], metrics: Dict[str, Any]) -> None:
    """写入输出。"""
    writer = FileWriter(output_path)
    writer.write(json.dumps(metrics, indent=2))


class TritonModelProfiler:
    """Triton 模型性能分析器，用于加载模型、执行推理并收集性能指标。"""

    def __init__(
        self,
        model_repository: str,
        model_name: str,
        input_data: str,
        protocol: str = "grpc",
        endpoint: Optional[str] = None,
        batch_size: int = 1,
        triton_server_path: Optional[str] = None,
        server_log: Optional[str] = None,
        output: Optional[str] = None,
        timeout: int = 5,
    ) -> None:
        """
        初始化模型分析器。

        Args:
            model_repository: Triton 模型仓库路径。
            model_name: 要分析的模型名称。
            input_data: 输入数据 JSON 文件或包含 JSON 文件的目录。
            protocol: 通信协议，'grpc' 或 'http'。
            endpoint: 服务器端点，如 'localhost:8001'。
            batch_size: 批处理大小。
            triton_server_path: Triton 服务器可执行文件路径（可选）。
            server_log: 服务器日志文件路径（可选）。
            output: 输出指标 JSON 文件路径（可选）。
            timeout: 等待服务器和模型就绪的超时时间（秒）。
        """
        self.model_repository = model_repository
        self.model_name = model_name
        self.input_data = input_data
        self.protocol = protocol
        self.endpoint = endpoint or get_default_endpoint(protocol, None)
        self.batch_size = batch_size
        self.triton_server_path = triton_server_path
        self.server_log = server_log
        self.output = output
        self.timeout = timeout

        self.server = None
        self.client = None
        self.model_config = None
        self._output_results = []

    def _setup_server(self) -> None:
        """设置并启动 Triton 服务器。"""
        server_config = create_server_config(self.model_repository, self.protocol, self.endpoint)
        self.server = TritonServerLocal(
            path=self.triton_server_path,
            config=server_config,
            gpus=[],
            log_path=self.server_log,
        )
        self.client = get_client(self.protocol, self.endpoint)
        self.client.wait_for_server_ready(num_retries=self.timeout)

    def _load_model(self) -> tuple:
        """
        加载模型并返回加载时间和内存信息。

        Returns:
            Tuple[float, float, float]: (model_load_time, model_load_memory, mem_after_load)
        """
        kytensor_proc = find_kytensor_process()
        mem_before_load, _ = self.server.cpu_stats(server_process=kytensor_proc)

        model_load_start = time.perf_counter()
        load_result = self.client.load_model(self.model_name)
        if load_result == -1:
            raise TritonModelAnalyzerException(f"加载模型 {self.model_name} 失败")
        self.client.wait_for_model_ready(self.model_name, num_retries=self.timeout)
        model_load_end = time.perf_counter()

        model_load_time = model_load_end - model_load_start
        mem_after_load, _ = self.server.cpu_stats(server_process=kytensor_proc)
        model_load_memory = mem_after_load - mem_before_load

        self.model_config = self.client.get_model_config(self.model_name, num_retries=self.timeout)

        return model_load_time, model_load_memory, mem_after_load

    def _run_inference(self, mem_after_load: float) -> tuple:
        """
        执行推理并收集性能指标。

        Args:
            mem_after_load: 模型加载后的内存使用量。

        Returns:
            Tuple[float, float, float, float, float, float, float, float, List[Any]]:
                (inference_time, avg_latency_ms, throughput, peak_memory,
                 server_cpu_uss, server_available, server_cpu_percent,
                 system_cpu_percent, output_results)
        """
        kytensor_proc = find_kytensor_process()
        input_payloads = load_input_payloads(self.input_data) if self.input_data else [None]
        num_requests = len(input_payloads)
        logger.info(f"使用 {num_requests} 个输入负载执行推理")

        infer_outputs = create_infer_outputs(self.model_config, self.protocol)

        server_cpu_uss, server_available = self.server.cpu_stats(server_process=kytensor_proc)

        peak_memory = mem_after_load
        inference_start = time.perf_counter()
        output_results = []

        for idx, payload in enumerate(input_payloads):
            infer_inputs = create_infer_inputs(self.model_config, self.batch_size, payload, self.protocol)
            result = self.client.infer(
                model_name=self.model_name,
                inputs=infer_inputs,
                outputs=infer_outputs,
                request_id=str(idx),
            )
            output_results.append(result)
            current_mem, _ = self.server.cpu_stats(server_process=kytensor_proc)
            if current_mem > peak_memory:
                peak_memory = current_mem

        inference_end = time.perf_counter()
        server_cpu_percent, system_cpu_percent = self.server.cpu_percent(interval=1, server_process=kytensor_proc)

        inference_time = inference_end - inference_start
        avg_latency_ms = inference_time * 1000.0 / num_requests if num_requests > 0 else 0.0
        throughput = num_requests / inference_time if inference_time > 0 else 0.0

        # 打印输出结果信息
        if output_results:
            last_result = output_results[-1]
            for output_config in self.model_config.get("output", []):
                output_name = output_config["name"]
                output_data = last_result.as_numpy(output_name)
                if output_data is not None:
                    if output_data.dtype == np.uint8:
                        try:
                            result_str = output_data.tobytes().decode("utf-8")
                            result_json = json.loads(result_str)
                            logger.info(f"输出 '{output_name}' 解析为 JSON: {result_json}")
                        except Exception:
                            output_data.tobytes().decode("utf-8", errors="ignore")
                    logger.info(f"输出 '{output_name}' 形状: {output_data.shape}, 数据类型: {output_data.dtype}")
                    logger.info(f"输出 '{output_name}' 样本数据: {output_data.flatten()[:10]}")

        return (
            inference_time,
            avg_latency_ms,
            throughput,
            peak_memory,
            server_cpu_uss,
            server_available,
            server_cpu_percent,
            system_cpu_percent,
            output_results,
        )

    def _unload_model(self) -> float:
        """
        卸载模型并返回卸载时间。

        Returns:
            model_unload_time
        """
        model_unload_start = time.perf_counter()
        unload_result = self.client.unload_model(self.model_name)
        if unload_result == -1:
            logger.warning(f"卸载模型 {self.model_name} 失败")
        model_unload_end = time.perf_counter()
        return model_unload_end - model_unload_start

    def _build_metrics(
        self,
        model_load_time: float,
        model_load_memory: float,
        model_unload_time: float,
        peak_memory: float,
        inference_time: float,
        avg_latency_ms: float,
        throughput: float,
        server_cpu_uss: float,
        server_available: float,
        server_cpu_percent: float,
        system_cpu_percent: float,
        num_requests: int,
    ) -> Dict[str, Any]:
        """构建性能指标字典。"""
        return {
            "model_name": self.model_name,
            "model_repository": os.path.abspath(self.model_repository),
            "protocol": self.protocol,
            "endpoint": self.endpoint,
            "batch_size": self.batch_size,
            "num_requests": num_requests,
            # 模型加载性能
            "model_load_time_s": model_load_time,
            "model_load_memory_mb": model_load_memory,
            # 模型卸载性能
            "model_unload_time_s": model_unload_time,
            # 推理性能
            "peak_inference_memory_mb": peak_memory,
            "inference_total_time_s": inference_time,
            "inference_average_latency_ms": avg_latency_ms,
            "inference_throughput_inferences_per_s": throughput,
            # 服务器资源
            "server_cpu_uss_mb": server_cpu_uss,
            "server_available_ram_mb": server_available,
            "server_cpu_percent": server_cpu_percent,
            "system_cpu_percent": system_cpu_percent,
        }

    def profile(self) -> Dict[str, Any]:
        """
        执行完整的性能分析流程。

        Returns:
            包含性能指标的字典。
        """
        try:
            # 设置服务器
            self._setup_server()

            # 加载模型
            model_load_time, model_load_memory, mem_after_load = self._load_model()

            # 执行推理
            (
                inference_time,
                avg_latency_ms,
                throughput,
                peak_memory,
                server_cpu_uss,
                server_available,
                server_cpu_percent,
                system_cpu_percent,
                output_results,
            ) = self._run_inference(mem_after_load)

            # 保存输出结果供外部访问
            self._output_results = output_results

            # 卸载模型
            model_unload_time = self._unload_model()

            # 计算请求数
            input_payloads = load_input_payloads(self.input_data) if self.input_data else [None]
            num_requests = len(input_payloads)

            # 构建指标
            metrics = self._build_metrics(
                model_load_time,
                model_load_memory,
                model_unload_time,
                peak_memory,
                inference_time,
                avg_latency_ms,
                throughput,
                server_cpu_uss,
                server_available,
                server_cpu_percent,
                system_cpu_percent,
                num_requests,
            )

            # 写入输出
            if self.output:
                output_dir = os.path.dirname(self.output)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                write_output(self.output, metrics)

            logger.info("分析完成。")
            return metrics

        except TritonModelAnalyzerException as e:
            logger.error(str(e))
            raise

    def get_output_results(self) -> List[Any]:
        """
        获取最后一次推理的输出结果。

        Returns:
            推理结果列表。
        """
        return self._output_results


# ============================================================
# CLI 命令行入口点
# ============================================================

import argparse as _argparse


def _parse_args():
    """解析命令行参数。"""
    parser = _argparse.ArgumentParser(description="Model Analyzer 测试脚本")
    parser.add_argument("--ocr-image-dir", type=str, default="test_data/ocr_image/",
                        help="OCR 模型原始图像数据目录")
    parser.add_argument("--gte-text-dir", type=str, default="test_data/gte_text/",
                        help="GTE 模型原始文本数据目录")
    parser.add_argument("--cn-clip-text-dir", type=str, default="test_data/cn_clip_text/",
                        help="CN-CLIP Text 模型原始文本数据目录")
    parser.add_argument("--cn-clip-image-dir", type=str, default="test_data/cn_clip_image/",
                        help="CN-CLIP Image 模型原始图像数据目录")
    parser.add_argument("--model-repository", type=str, default="/usr/share/kylin-ai/model-repository",
                        help="Triton 模型仓库路径")
    parser.add_argument("--triton-endpoint", type=str, default="localhost:8001",
                        help="Triton 服务器地址")
    parser.add_argument("--skip-ocr", action="store_true", help="跳过 OCR 模型测试")
    parser.add_argument("--skip-gte", action="store_true", help="跳过 GTE 模型测试")
    parser.add_argument("--skip-cn-clip-text", action="store_true", help="跳过 CN-CLIP Text 模型测试")
    parser.add_argument("--skip-cn-clip-image", action="store_true", help="跳过 CN-CLIP Image 模型测试")
    return parser.parse_args()


def _test_ocr_model(args):
    """测试 OCR 模型。"""
    try:
        from model_analyzer.process_data import DataProcessorFactory
    except ImportError:
        from process_data import DataProcessorFactory

    print("=" * 60)
    print("测试 OCR 模型 (ocr_ppocr)")
    print("=" * 60)

    ocr_processor = DataProcessorFactory.create_processor("ocr_ppocr", output_dir="input_data")
    ocr_output_files = ocr_processor.process_directory(args.ocr_image_dir)

    ocr_profiler = TritonModelProfiler(
        model_repository=args.model_repository,
        model_name="ocr_ppocr",
        input_data="input_data/",
        protocol="grpc",
        endpoint=args.triton_endpoint,
        output="metrics_ocr.json",
    )

    ocr_metrics = ocr_profiler.profile()
    print(f"OCR 性能指标: {ocr_metrics}")
    return ocr_metrics


def _test_gte_model(args):
    """测试 GTE 模型。"""
    try:
        from model_analyzer.process_data import DataProcessorFactory
    except ImportError:
        from process_data import DataProcessorFactory

    print("\n" + "=" * 60)
    print("测试 GTE 模型 (embd_gte-base_uint8-text)")
    print("=" * 60)

    gte_processor = DataProcessorFactory.create_processor(
        "embd_gte-base_uint8-text",
        output_dir="input_data_gte"
    )

    gte_output_files = gte_processor.process_directory(args.gte_text_dir)

    print(f"\n为 GTE 模型生成 {len(gte_output_files)} 个输入文件:")
    for f in gte_output_files:
        print(f"  - {f}")

    gte_profiler = TritonModelProfiler(
        model_repository=args.model_repository,
        model_name="embd_gte-base_uint8-text",
        input_data="input_data_gte/",
        protocol="grpc",
        endpoint=args.triton_endpoint,
        output="metrics_gte.json",
    )

    gte_metrics = gte_profiler.profile()
    gte_output_results = gte_profiler.get_output_results()

    print(f"\nGTE 性能指标: {gte_metrics}")

    if gte_output_results:
        print("\nGTE 输出嵌入信息:")
        for i, result in enumerate(gte_output_results):
            for output_config in gte_profiler.model_config.get("output", []):
                output_name = output_config["name"]
                output_data = result.as_numpy(output_name)
                if output_data is not None:
                    print(f"  样本 {i}: 输出 '{output_name}' 形状={output_data.shape}, 数据类型={output_data.dtype}")

    return gte_metrics


def _test_cn_clip_text_model(args):
    """测试 CN-CLIP Text 模型。"""
    try:
        from model_analyzer.process_data import DataProcessorFactory
    except ImportError:
        from process_data import DataProcessorFactory

    print("\n" + "=" * 60)
    print("测试 CN-CLIP Text 模型 (embd_cn-clip_512-uint8-text)")
    print("=" * 60)

    cn_clip_text_processor = DataProcessorFactory.create_processor(
        "embd_cn-clip_512-uint8-text",
        output_dir="input_data_cn_clip_text"
    )

    cn_clip_text_output_files = cn_clip_text_processor.process_directory(args.cn_clip_text_dir)

    print(f"\n为 CN-CLIP Text 模型生成 {len(cn_clip_text_output_files)} 个输入文件:")
    for f in cn_clip_text_output_files:
        print(f"  - {f}")

    cn_clip_text_profiler = TritonModelProfiler(
        model_repository=args.model_repository,
        model_name="embd_cn-clip_512-uint8-text",
        input_data="input_data_cn_clip_text/",
        protocol="grpc",
        endpoint=args.triton_endpoint,
        output="metrics_cn_clip_text.json",
    )

    cn_clip_text_metrics = cn_clip_text_profiler.profile()
    cn_clip_text_output_results = cn_clip_text_profiler.get_output_results()

    print(f"\nCN-CLIP Text 性能指标: {cn_clip_text_metrics}")

    if cn_clip_text_output_results:
        print("\nCN-CLIP Text 输出嵌入信息:")
        for i, result in enumerate(cn_clip_text_output_results):
            for output_config in cn_clip_text_profiler.model_config.get("output", []):
                output_name = output_config["name"]
                output_data = result.as_numpy(output_name)
                if output_data is not None:
                    print(f"  样本 {i}: 输出 '{output_name}' 形状={output_data.shape}, 数据类型={output_data.dtype}")

    return cn_clip_text_metrics


def _test_cn_clip_image_model(args):
    """测试 CN-CLIP Image 模型。"""
    try:
        from model_analyzer.process_data import DataProcessorFactory
    except ImportError:
        from process_data import DataProcessorFactory

    print("\n" + "=" * 60)
    print("测试 CN-CLIP Image 模型 (embd_cn-clip_512-uint8-image)")
    print("=" * 60)

    cn_clip_image_processor = DataProcessorFactory.create_processor(
        "embd_cn-clip_512-uint8-image",
        output_dir="input_data_cn_clip_image"
    )

    cn_clip_image_output_files = cn_clip_image_processor.process_directory(args.cn_clip_image_dir)

    print(f"\n为 CN-CLIP Image 模型生成 {len(cn_clip_image_output_files)} 个输入文件:")
    for f in cn_clip_image_output_files:
        print(f"  - {f}")

    cn_clip_image_profiler = TritonModelProfiler(
        model_repository=args.model_repository,
        model_name="embd_cn-clip_512-uint8-image",
        input_data="input_data_cn_clip_image/",
        protocol="grpc",
        endpoint=args.triton_endpoint,
        output="metrics_cn_clip_image.json",
    )

    cn_clip_image_metrics = cn_clip_image_profiler.profile()
    cn_clip_image_output_results = cn_clip_image_profiler.get_output_results()

    print(f"\nCN-CLIP Image 性能指标: {cn_clip_image_metrics}")

    if cn_clip_image_output_results:
        print("\nCN-CLIP Image 输出嵌入信息:")
        for i, result in enumerate(cn_clip_image_output_results):
            for output_config in cn_clip_image_profiler.model_config.get("output", []):
                output_name = output_config["name"]
                output_data = result.as_numpy(output_name)
                if output_data is not None:
                    print(f"  样本 {i}: 输出 '{output_name}' 形状={output_data.shape}, 数据类型={output_data.dtype}")

    return cn_clip_image_metrics


def main():
    """model-analyzer 命令行入口点。"""
    args = _parse_args()

    print(f"模型仓库: {args.model_repository}")
    print(f"Triton 端点: {args.triton_endpoint}")
    print()

    if not args.skip_ocr:
        try:
            _test_ocr_model(args)
        except Exception as e:
            print(f"OCR 模型测试失败: {e}")

    if not args.skip_gte:
        try:
            _test_gte_model(args)
        except Exception as e:
            print(f"GTE 模型测试失败: {e}")

    if not args.skip_cn_clip_text:
        try:
            _test_cn_clip_text_model(args)
        except Exception as e:
            print(f"CN-CLIP Text 模型测试失败: {e}")

    if not args.skip_cn_clip_image:
        try:
            _test_cn_clip_image_model(args)
        except Exception as e:
            print(f"CN-CLIP Image 模型测试失败: {e}")

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)