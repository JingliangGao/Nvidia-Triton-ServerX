#!/usr/bin/env python3

from abc import ABC, abstractmethod
from io import TextIOWrapper


class TritonServer(ABC):
    """
    定义由 TritonServerFactory 创建的对象的接口基类。
    """

    @abstractmethod
    def start(self, env=None):
        """
        启动 tritonserver。

        Parameters
        ----------
        env: dict
            用于此次 tritonserver 启动的环境变量。
        """

    @abstractmethod
    def stop(self):
        """
        停止并清理服务器资源。
        """

    @abstractmethod
    def log_file(self) -> TextIOWrapper:
        """
        返回服务器的日志文件。
        """

    @abstractmethod
    def cpu_stats(self):
        """
        返回 CPU 内存使用量和可用内存（MB）。
        """

    def update_config(self, params):
        """
        更新服务器参数。

        Parameters
        ----------
        params: dict
            键为参数名，值为参数值。
        """

        self._server_config.update_config(params)

    def config(self):
        """
        Returns
        -------
        TritonServerConfig
            当前服务器的配置。
        """

        return self._server_config