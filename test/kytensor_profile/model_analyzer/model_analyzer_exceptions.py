#!/usr/bin/env python3


class TritonModelAnalyzerException(Exception):
    """
    专用于 Triton Model Analyzer 的自定义异常类。
    """

    def __init__(self, message: str) -> None:
        """
        初始化异常对象。

        Args:
            message: 错误信息。
        """
        super().__init__(message)
        self.message = message