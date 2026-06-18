#!/usr/bin/env python3

from .grpc_client import TritonGRPCClient
from .http_client import TritonHTTPClient


class TritonClientFactory:
    """
    客户端工厂基类，用于创建不同类型的 Triton 客户端。
    """

    @staticmethod
    def create_grpc_client(server_url, ssl_options={}):
        """
        创建 gRPC 客户端。

        Parameters
        ----------
        server_url : str
            Triton 服务器的 gRPC 端点 URL。
        ssl_options : dict
            gRPC Python 客户端的 SSL 选项字典。

        Returns
        -------
        TritonGRPCClient
            gRPC 客户端实例。
        """
        return TritonGRPCClient(server_url=server_url, ssl_options=ssl_options)

    @staticmethod
    def create_http_client(server_url, ssl_options={}):
        """
        创建 HTTP 客户端。

        Parameters
        ----------
        server_url : str
            Triton 服务器的 HTTP 端点 URL。
        ssl_options : dict
            HTTP Python 客户端的 SSL 选项字典。

        Returns
        -------
        TritonHTTPClient
            HTTP 客户端实例。
        """
        return TritonHTTPClient(server_url=server_url, ssl_options=ssl_options)