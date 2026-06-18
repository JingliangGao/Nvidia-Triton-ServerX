#!/usr/bin/env python3

import os
import sys
import json
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
import numpy as np
from PIL import Image

# 使用 HuggingFace 镜像以应对网络问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None


class DataProcessor(ABC):
    """数据处理基类，定义数据处理接口。"""

    def __init__(self, output_dir: str = "input_data"):
        """
        初始化数据处理器。

        Args:
            output_dir: 输出 JSON 文件的目录。
        """
        self.output_dir = output_dir

    @abstractmethod
    def preprocess(self, image_path: str) -> Dict[str, Any]:
        """预处理输入数据。"""
        pass

    def process_and_save(self, image_path: str) -> str:
        """
        处理并保存输入数据为 JSON 文件。

        Args:
            image_path: 输入文件路径。

        Returns:
            输出的 JSON 文件路径。
        """
        payload = self.preprocess(image_path)
        os.makedirs(self.output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_file = os.path.join(self.output_dir, f"{base_name}.json")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        return output_file

    def process_directory(self, test_dir: str) -> List[str]:
        """
        处理目录中的所有输入数据。

        Args:
            test_dir: 输入数据目录路径。

        Returns:
            输出的 JSON 文件路径列表。
        """
        supported_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".txt"}

        if not os.path.isdir(test_dir):
            raise ValueError(f"目录 '{test_dir}' 不存在或不是一个目录。")

        data_files = sorted([
            f for f in os.listdir(test_dir)
            if os.path.splitext(f.lower())[1] in supported_extensions
        ])

        if not data_files:
            print(f"警告: 在 '{test_dir}' 中未找到支持的输入文件。")
            return []

        output_files = []
        for data_name in data_files:
            data_path = os.path.join(test_dir, data_name)
            try:
                output_file = self.process_and_save(data_path)
                output_files.append(output_file)
                print(f"已处理: {data_name} -> {output_file}")
            except Exception as e:
                print(f"处理 '{data_name}' 时出错: {e}")

        print(f"\n成功保存 {len(output_files)} 个 JSON 文件到 '{self.output_dir}'。")
        return output_files


class OCRDataProcessor(DataProcessor):
    """OCR 模型数据处理器。"""

    def preprocess(self, image_path: str) -> Dict[str, Any]:
        """
        预处理 OCR 模型输入图像。

        Args:
            image_path: 图像文件路径。

        Returns:
            包含 IN0 输入数据的字典。
        """
        img_array = self._load_image(image_path)
        return {
            "IN0": img_array.tolist()
        }

    def _load_image(self, image_path: str) -> np.ndarray:
        """
        加载并预处理图像。

        Args:
            image_path: 图像文件路径。

        Returns:
            BGR 3 通道 numpy 数组，形状为 [H, W, C]。
        """
        img = Image.open(image_path)

        # 首先转换为 RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        # 转换为 numpy 数组
        img_array = np.array(img, dtype=np.uint8)

        return img_array


class SimpleTokenizer:
    """简单的 UTF-8 字符级 tokenizer，用于无法下载 HuggingFace 模型时的回退方案。"""

    def __init__(self):
        # 构建简单的字符到 ID 映射
        self.vocab = {}
        # 特殊 tokens
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.cls_token_id = 2
        self.sep_token_id = 3
        # 为常见字符分配 ID（从 4 开始）
        self._build_vocab()

    def _build_vocab(self):
        """构建简单词汇表。"""
        # ASCII 可打印字符
        for i in range(32, 127):
            self.vocab[chr(i)] = len(self.vocab) + 4
        # 常见 Unicode 范围（CJK, 韩文, 日文等）
        for codepoint in range(0x4E00, 0x9FFF + 1):  # CJK 统一汉字
            char = chr(codepoint)
            self.vocab[char] = len(self.vocab) + 4
        for codepoint in range(0x3040, 0x309F + 1):  # 平假名
            char = chr(codepoint)
            self.vocab[char] = len(self.vocab) + 4
        for codepoint in range(0x30A0, 0x30FF + 1):  # 片假名
            char = chr(codepoint)
            self.vocab[char] = len(self.vocab) + 4
        for codepoint in range(0xAC00, 0xD7AF + 1):  # 韩文音节
            char = chr(codepoint)
            self.vocab[char] = len(self.vocab) + 4
        for codepoint in range(0x00C0, 0x024F + 1):  # 拉丁扩展
            char = chr(codepoint)
            self.vocab[char] = len(self.vocab) + 4

    def encode(self, text: str, truncation: bool = True, max_length: int = 512, **kwargs) -> Dict[str, np.ndarray]:
        """简单编码文本。"""
        # 添加 CLS token
        input_ids = [self.cls_token_id]
        for char in text:
            token_id = self.vocab.get(char, self.unk_token_id)
            input_ids.append(token_id)
        # 添加 SEP token
        input_ids.append(self.sep_token_id)

        # 截断
        if truncation and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            input_ids[-1] = self.sep_token_id

        # 创建注意力掩码（1 表示真实 token，0 表示填充）
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": np.array(input_ids, dtype=np.int64),
            "attention_mask": np.array(attention_mask, dtype=np.int64)
        }


class GTETextDataProcessor(DataProcessor):
    """GTE（General Text Embedding）模型数据处理器。"""

    def __init__(self, output_dir: str = "input_data", model_name: str = "Alibaba-NLP/gte-multilingual-base"):
        """
        初始化 GTE 数据处理器。

        Args:
            output_dir: 输出 JSON 文件的目录。
            model_name: HuggingFace tokenizer 模型名称（如果无法下载则使用简单 tokenizer）。
        """
        super().__init__(output_dir)
        self.tokenizer = None
        self.use_simple_tokenizer = False

        if AutoTokenizer is not None:
            try:
                # 使用镜像以应对网络问题
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    resume_download=True
                )
            except Exception as e:
                print(f"警告: 无法加载 HuggingFace tokenizer '{model_name}': {e}")
                print("使用简单字符级 tokenizer 作为回退方案。")
                self.use_simple_tokenizer = True

        if self.tokenizer is None:
            self.tokenizer = SimpleTokenizer()
            self.use_simple_tokenizer = True

    def preprocess(self, text_path: str) -> Dict[str, Any]:
        """
        预处理文本文件，生成 GTE 模型输入。

        Args:
            text_path: 文本文件路径。

        Returns:
            包含 input_ids 和 attention_mask 的字典。
        """
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            raise ValueError(f"文本文件 '{text_path}' 为空。")

        if self.use_simple_tokenizer:
            encoded = self.tokenizer.encode(
                text,
                truncation=True,
                max_length=512
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
        else:
            # 使用 HuggingFace tokenizer 对文本进行分词
            encoded = self.tokenizer(
                text,
                truncation=True,
                max_length=512,
                padding=False,
                return_tensors="np"
            )
            input_ids = encoded["input_ids"][0].astype(np.int64)
            attention_mask = encoded["attention_mask"][0].astype(np.int64)

        return {
            "input_ids": input_ids.tolist(),
            "attention_mask": attention_mask.tolist()
        }

    def process_and_save(self, text_path: str) -> str:
        """
        处理文本文件并保存为 JSON 文件。

        Args:
            text_path: 文本文件路径。

        Returns:
            输出的 JSON 文件路径。
        """
        payload = self.preprocess(text_path)
        os.makedirs(self.output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(text_path))[0]
        output_file = os.path.join(self.output_dir, f"{base_name}.json")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        return output_file

    def process_directory(self, test_dir: str) -> List[str]:
        """
        处理目录中的所有文本文件。

        Args:
            test_dir: 文本目录路径。

        Returns:
            输出的 JSON 文件路径列表。
        """
        supported_extensions = {".txt"}

        if not os.path.isdir(test_dir):
            raise ValueError(f"目录 '{test_dir}' 不存在或不是一个目录。")

        text_files = sorted([
            f for f in os.listdir(test_dir)
            if os.path.splitext(f.lower())[1] in supported_extensions
        ])

        if not text_files:
            print(f"警告: 在 '{test_dir}' 中未找到文本文件。")
            return []

        output_files = []
        for text_name in text_files:
            text_path = os.path.join(test_dir, text_name)
            try:
                output_file = self.process_and_save(text_path)
                output_files.append(output_file)
                print(f"已处理: {text_name} -> {output_file}")
            except Exception as e:
                print(f"处理 '{text_name}' 时出错: {e}")

        print(f"\n成功保存 {len(output_files)} 个 JSON 文件到 '{self.output_dir}'。")
        return output_files


class CNCLIPTextDataProcessor(DataProcessor):
    """CN-CLIP Text 模型数据处理器。"""

    def __init__(self, output_dir: str = "input_data", model_name: str = "OFA-Sys/chinese-clip-vit-base-patch16"):
        """
        初始化 CN-CLIP 数据处理器。

        Args:
            output_dir: 输出 JSON 文件的目录。
            model_name: HuggingFace tokenizer 模型名称。
        """
        super().__init__(output_dir)
        self.tokenizer = None
        self.use_simple_tokenizer = False

        if AutoTokenizer is not None:
            try:
                # 使用镜像以应对网络问题
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    resume_download=True
                )
            except Exception as e:
                print(f"警告: 无法加载 HuggingFace tokenizer '{model_name}': {e}")
                print("使用简单字符级 tokenizer 作为回退方案。")
                self.use_simple_tokenizer = True

        if self.tokenizer is None:
            self.tokenizer = SimpleTokenizer()
            self.use_simple_tokenizer = True

    def preprocess(self, text_path: str) -> Dict[str, Any]:
        """
        预处理文本文件，生成 CN-CLIP 模型输入。

        Args:
            text_path: 文本文件路径。

        Returns:
            包含 text 输入的字典（形状: [512]）。
        """
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            raise ValueError(f"文本文件 '{text_path}' 为空。")

        if self.use_simple_tokenizer:
            encoded = self.tokenizer.encode(
                text,
                truncation=True,
                max_length=512
            )
            input_ids = encoded["input_ids"]
        else:
            # 使用 HuggingFace tokenizer 对文本进行分词
            encoded = self.tokenizer(
                text,
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="np"
            )
            input_ids = encoded["input_ids"][0].astype(np.int64)

        # 确保输出形状为 [512]
        if len(input_ids) < 512:
            # 如果短于 512，用 0 填充
            input_ids = np.pad(input_ids, (0, 512 - len(input_ids)), constant_values=0)
        elif len(input_ids) > 512:
            # 如果长于 512，截断
            input_ids = input_ids[:512]

        return {
            "text": input_ids.tolist()
        }

    def process_and_save(self, text_path: str) -> str:
        """
        处理文本文件并保存为 JSON 文件。

        Args:
            text_path: 文本文件路径。

        Returns:
            输出的 JSON 文件路径。
        """
        payload = self.preprocess(text_path)
        os.makedirs(self.output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(text_path))[0]
        output_file = os.path.join(self.output_dir, f"{base_name}.json")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        return output_file

    def process_directory(self, test_dir: str) -> List[str]:
        """
        处理目录中的所有文本文件。

        Args:
            test_dir: 文本目录路径。

        Returns:
            输出的 JSON 文件路径列表。
        """
        supported_extensions = {".txt"}

        if not os.path.isdir(test_dir):
            raise ValueError(f"目录 '{test_dir}' 不存在或不是一个目录。")

        text_files = sorted([
            f for f in os.listdir(test_dir)
            if os.path.splitext(f.lower())[1] in supported_extensions
        ])

        if not text_files:
            print(f"警告: 在 '{test_dir}' 中未找到文本文件。")
            return []

        output_files = []
        for text_name in text_files:
            text_path = os.path.join(test_dir, text_name)
            try:
                output_file = self.process_and_save(text_path)
                output_files.append(output_file)
                print(f"已处理: {text_name} -> {output_file}")
            except Exception as e:
                print(f"处理 '{text_name}' 时出错: {e}")

        print(f"\n成功保存 {len(output_files)} 个 JSON 文件到 '{self.output_dir}'。")
        return output_files


class CNCLIPImageDataProcessor(DataProcessor):
    """CN-CLIP Image 模型数据处理器。"""

    def __init__(
        self,
        output_dir: str = "input_data",
        image_size: int = 224,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ):
        """
        初始化 CN-CLIP Image 数据处理器。

        Args:
            output_dir: 输出 JSON 文件的目录。
            image_size: 图像 resize 尺寸。
            mean: 归一化均值，默认使用 CLIP 标准值。
            std: 归一化标准差，默认使用 CLIP 标准值。
        """
        super().__init__(output_dir)
        self.image_size = image_size
        self.mean = mean if mean is not None else [0.48145466, 0.4578275, 0.40821073]
        self.std = std if std is not None else [0.26862954, 0.26130258, 0.27577711]

    def preprocess(self, image_path: str) -> Dict[str, Any]:
        """
        预处理图像文件，生成 CN-CLIP Image 模型输入。

        Args:
            image_path: 图像文件路径。

        Returns:
            包含 image 输入的字典（形状: [3, 224, 224]，NCHW 格式，FP32）。
        """
        img_array = self._load_and_preprocess_image(image_path)
        return {
            "image": img_array.tolist()
        }

    def _load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        """
        加载并预处理图像为 CN-CLIP 模型输入格式。

        Args:
            image_path: 图像文件路径。

        Returns:
            NCHW 格式的 FP32 numpy 数组，形状为 [3, 224, 224]。
        """
        # 打开图像并转换为 RGB
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize 到目标尺寸
        img = img.resize((self.image_size, self.image_size), Image.BICUBIC)

        # 转换为 numpy 数组并归一化到 [0, 1]
        img_array = np.array(img, dtype=np.float32) / 255.0

        # 使用均值和标准差进行归一化
        mean = np.array(self.mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(self.std, dtype=np.float32).reshape(1, 1, 3)
        img_array = (img_array - mean) / std

        # 转换 HWC -> NCHW（去掉 batch 维度，因为模型配置中 reshape 会添加）
        # HWC [224, 224, 3] -> CHW [3, 224, 224]
        img_array = img_array.transpose(2, 0, 1)

        return img_array


class DataProcessorFactory:
    """数据处理器工厂类，根据模型名创建对应的处理器。"""

    _processors = {
        "ocr": OCRDataProcessor,
        "ocr_ppocr": OCRDataProcessor,
        "gte": GTETextDataProcessor,
        "embd_gte-base_uint8-text": GTETextDataProcessor,
        "cn-clip": CNCLIPTextDataProcessor,
        "embd_cn-clip_512-uint8-text": CNCLIPTextDataProcessor,
        "cn-clip-image": CNCLIPImageDataProcessor,
        "embd_cn-clip_512-uint8-image": CNCLIPImageDataProcessor,
    }

    @classmethod
    def register_processor(cls, model_name: str, processor_class: type) -> None:
        """
        注册新的数据处理器。

        Args:
            model_name: 模型名称。
            processor_class: 处理器类。
        """
        cls._processors[model_name] = processor_class

    @classmethod
    def create_processor(cls, model_name: str, output_dir: str = "input_data") -> DataProcessor:
        """
        根据模型名创建对应的数据处理器。

        Args:
            model_name: 模型名称。
            output_dir: 输出目录。

        Returns:
            数据处理器实例。

        Raises:
            ValueError: 如果模型名没有对应的处理器。
        """
        # 尝试精确匹配
        if model_name in cls._processors:
            return cls._processors[model_name](output_dir)

        # 尝试模糊匹配（检查模型名是否包含已知的处理器键）
        for key, processor_class in cls._processors.items():
            if key in model_name.lower():
                return processor_class(output_dir)

        raise ValueError(
            f"未找到模型 '{model_name}' 对应的数据处理器。"
            f"可用的处理器: {list(cls._processors.keys())}。"
            f"使用 DataProcessorFactory.register_processor() 添加新的处理器。"
        )


def load_image(image_path: str) -> np.ndarray:
    """
    加载图像的便捷函数。

    Args:
        image_path: 图像文件路径。

    Returns:
        BGR 3 通道 numpy 数组，形状为 [H, W, C]。
    """
    processor = OCRDataProcessor()
    return processor._load_image(image_path)


def process_test_images(test_dir: str, output_dir: str = "input_data", model_name: str = "ocr") -> List[str]:
    """
    处理测试图像的便捷函数。

    Args:
        test_dir: 测试图像目录。
        output_dir: 输出目录。
        model_name: 模型名，用于选择对应的处理器。

    Returns:
        输出的 JSON 文件路径列表。
    """
    processor = DataProcessorFactory.create_processor(model_name, output_dir)
    return processor.process_directory(test_dir)