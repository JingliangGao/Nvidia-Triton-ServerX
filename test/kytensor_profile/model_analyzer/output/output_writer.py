#!/usr/bin/env python3

from abc import ABC, abstractmethod


class OutputWriter(ABC):
    """
    输出写入器接口基类，接收输出内容并写入文件或流。
    """

    @abstractmethod
    def write(self, out):
        """
        将输出写入文件（标准输出、.txt、.csv 等）。
        """
