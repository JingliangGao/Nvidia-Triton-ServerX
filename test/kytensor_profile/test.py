#!/usr/bin/env python3
"""
Model Analyzer 测试脚本

用法:
    python test.py [选项]

选项:
    --ocr-image-dir DIR          OCR模型原始图像数据目录 (默认: test_data/ocr_image/)
    --gte-text-dir DIR           GTE模型原始文本数据目录 (默认: test_data/gte_text/)
    --cn-clip-text-dir DIR       CN-CLIP Text模型原始文本数据目录 (默认: test_data/cn_clip_text/)
    --cn-clip-image-dir DIR      CN-CLIP Image模型原始图像数据目录 (默认: test_data/cn_clip_image/)
    --model-repository DIR       Triton模型仓库路径 (默认: /usr/share/kylin-ai/model-repository)
    --triton-endpoint ENDPOINT   Triton服务器地址 (默认: localhost:8001)
    --skip-ocr                   跳过OCR模型测试
    --skip-gte                   跳过GTE模型测试
    --skip-cn-clip-text          跳过CN-CLIP Text模型测试
    --skip-cn-clip-image         跳过CN-CLIP Image模型测试
"""

import argparse
import sys
from model_analyzer import TritonModelProfiler, DataProcessorFactory


def parse_args():
    parser = argparse.ArgumentParser(description="Model Analyzer 测试脚本")
    parser.add_argument("--ocr-image-dir", type=str, default="test_data/ocr_image/",
                        help="OCR模型原始图像数据目录")
    parser.add_argument("--gte-text-dir", type=str, default="test_data/gte_text/",
                        help="GTE模型原始文本数据目录")
    parser.add_argument("--cn-clip-text-dir", type=str, default="test_data/cn_clip_text/",
                        help="CN-CLIP Text模型原始文本数据目录")
    parser.add_argument("--cn-clip-image-dir", type=str, default="test_data/cn_clip_image/",
                        help="CN-CLIP Image模型原始图像数据目录")
    parser.add_argument("--model-repository", type=str, default="/usr/share/kylin-ai/model-repository",
                        help="Triton模型仓库路径")
    parser.add_argument("--triton-endpoint", type=str, default="localhost:8001",
                        help="Triton服务器地址")
    parser.add_argument("--skip-ocr", action="store_true", help="跳过OCR模型测试")
    parser.add_argument("--skip-gte", action="store_true", help="跳过GTE模型测试")
    parser.add_argument("--skip-cn-clip-text", action="store_true", help="跳过CN-CLIP Text模型测试")
    parser.add_argument("--skip-cn-clip-image", action="store_true", help="跳过CN-CLIP Image模型测试")
    return parser.parse_args()


def test_ocr_model(args):
    """测试OCR模型"""
    print("=" * 60)
    print("Testing OCR Model (ocr_ppocr)")
    print("=" * 60)

    # 使用工厂创建处理器
    ocr_processor = DataProcessorFactory.create_processor("ocr_ppocr", output_dir="input_data")

    # 处理原始图像数据目录
    ocr_output_files = ocr_processor.process_directory(args.ocr_image_dir)

    ocr_profiler = TritonModelProfiler(
        model_repository=args.model_repository,
        model_name="ocr_ppocr",
        input_data="input_data/",
        protocol="grpc",
        endpoint=args.triton_endpoint,
        output="metrics_ocr.json",
    )

    # 执行性能分析
    ocr_metrics = ocr_profiler.profile()

    # 获取推理输出结果
    ocr_output_results = ocr_profiler.get_output_results()

    print(f"OCR Performance Metrics: {ocr_metrics}")
    return ocr_metrics


def test_gte_model(args):
    """测试GTE模型"""
    print("\n" + "=" * 60)
    print("Testing GTE Model (embd_gte-base_uint8-text)")
    print("=" * 60)

    # 使用工厂创建GTE文本数据处理器
    gte_processor = DataProcessorFactory.create_processor(
        "embd_gte-base_uint8-text",
        output_dir="input_data_gte"
    )

    # 处理原始文本数据目录
    gte_output_files = gte_processor.process_directory(args.gte_text_dir)

    print(f"\nGenerated {len(gte_output_files)} input files for GTE model:")
    for f in gte_output_files:
        print(f"  - {f}")

    # 创建GTE模型性能分析器
    gte_profiler = TritonModelProfiler(
        model_repository=args.model_repository,
        model_name="embd_gte-base_uint8-text",
        input_data="input_data_gte/",
        protocol="grpc",
        endpoint=args.triton_endpoint,
        output="metrics_gte.json",
    )

    # 执行性能分析
    gte_metrics = gte_profiler.profile()

    # 获取推理输出结果
    gte_output_results = gte_profiler.get_output_results()

    print(f"\nGTE Performance Metrics: {gte_metrics}")

    # 打印嵌入向量信息
    if gte_output_results:
        print("\nGTE Output Embedding Info:")
        for i, result in enumerate(gte_output_results):
            for output_config in gte_profiler.model_config.get("output", []):
                output_name = output_config["name"]
                output_data = result.as_numpy(output_name)
                if output_data is not None:
                    print(f"  Sample {i}: Output '{output_name}' shape={output_data.shape}, dtype={output_data.dtype}")

    return gte_metrics


def test_cn_clip_text_model(args):
    """测试CN-CLIP Text模型"""
    print("\n" + "=" * 60)
    print("Testing CN-CLIP Text Model (embd_cn-clip_512-uint8-text)")
    print("=" * 60)

    # 使用工厂创建CN-CLIP Text数据处理器
    cn_clip_text_processor = DataProcessorFactory.create_processor(
        "embd_cn-clip_512-uint8-text",
        output_dir="input_data_cn_clip_text"
    )

    # 处理原始文本数据目录
    cn_clip_text_output_files = cn_clip_text_processor.process_directory(args.cn_clip_text_dir)

    print(f"\nGenerated {len(cn_clip_text_output_files)} input files for CN-CLIP Text model:")
    for f in cn_clip_text_output_files:
        print(f"  - {f}")

    # 创建CN-CLIP Text模型性能分析器
    cn_clip_text_profiler = TritonModelProfiler(
        model_repository=args.model_repository,
        model_name="embd_cn-clip_512-uint8-text",
        input_data="input_data_cn_clip_text/",
        protocol="grpc",
        endpoint=args.triton_endpoint,
        output="metrics_cn_clip_text.json",
    )

    # 执行性能分析
    cn_clip_text_metrics = cn_clip_text_profiler.profile()

    # 获取推理输出结果
    cn_clip_text_output_results = cn_clip_text_profiler.get_output_results()

    print(f"\nCN-CLIP Text Performance Metrics: {cn_clip_text_metrics}")

    # 打印输出向量信息
    if cn_clip_text_output_results:
        print("\nCN-CLIP Text Output Embedding Info:")
        for i, result in enumerate(cn_clip_text_output_results):
            for output_config in cn_clip_text_profiler.model_config.get("output", []):
                output_name = output_config["name"]
                output_data = result.as_numpy(output_name)
                if output_data is not None:
                    print(f"  Sample {i}: Output '{output_name}' shape={output_data.shape}, dtype={output_data.dtype}")

    return cn_clip_text_metrics


def test_cn_clip_image_model(args):
    """测试CN-CLIP Image模型"""
    print("\n" + "=" * 60)
    print("Testing CN-CLIP Image Model (embd_cn-clip_512-uint8-image)")
    print("=" * 60)

    # 使用工厂创建CN-CLIP Image数据处理器
    cn_clip_image_processor = DataProcessorFactory.create_processor(
        "embd_cn-clip_512-uint8-image",
        output_dir="input_data_cn_clip_image"
    )

    # 处理原始图像数据目录
    cn_clip_image_output_files = cn_clip_image_processor.process_directory(args.cn_clip_image_dir)

    print(f"\nGenerated {len(cn_clip_image_output_files)} input files for CN-CLIP Image model:")
    for f in cn_clip_image_output_files:
        print(f"  - {f}")

    # 创建CN-CLIP Image模型性能分析器
    cn_clip_image_profiler = TritonModelProfiler(
        model_repository=args.model_repository,
        model_name="embd_cn-clip_512-uint8-image",
        input_data="input_data_cn_clip_image/",
        protocol="grpc",
        endpoint=args.triton_endpoint,
        output="metrics_cn_clip_image.json",
    )

    # 执行性能分析
    cn_clip_image_metrics = cn_clip_image_profiler.profile()

    # 获取推理输出结果
    cn_clip_image_output_results = cn_clip_image_profiler.get_output_results()

    print(f"\nCN-CLIP Image Performance Metrics: {cn_clip_image_metrics}")

    # 打印输出向量信息
    if cn_clip_image_output_results:
        print("\nCN-CLIP Image Output Embedding Info:")
        for i, result in enumerate(cn_clip_image_output_results):
            for output_config in cn_clip_image_profiler.model_config.get("output", []):
                output_name = output_config["name"]
                output_data = result.as_numpy(output_name)
                if output_data is not None:
                    print(f"  Sample {i}: Output '{output_name}' shape={output_data.shape}, dtype={output_data.dtype}")

    return cn_clip_image_metrics


def main():
    args = parse_args()

    print(f"Model Repository: {args.model_repository}")
    print(f"Triton Endpoint: {args.triton_endpoint}")
    print()

    # 测试OCR模型
    if not args.skip_ocr:
        try:
            test_ocr_model(args)
        except Exception as e:
            print(f"OCR model test failed: {e}")

    # 测试GTE模型
    if not args.skip_gte:
        try:
            test_gte_model(args)
        except Exception as e:
            print(f"GTE model test failed: {e}")

    # 测试CN-CLIP Text模型
    if not args.skip_cn_clip_text:
        try:
            test_cn_clip_text_model(args)
        except Exception as e:
            print(f"CN-CLIP Text model test failed: {e}")

    # 测试CN-CLIP Image模型
    if not args.skip_cn_clip_image:
        try:
            test_cn_clip_image_model(args)
        except Exception as e:
            print(f"CN-CLIP Image model test failed: {e}")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()