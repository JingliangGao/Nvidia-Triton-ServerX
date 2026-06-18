#!/usr/bin/env python3

import tritonclient.grpc as grpcclient

from .client import TritonClient


class TritonGRPCClient(TritonClient):
    """
    TritonClient 的 gRPC 具体实现。
    """

    def __init__(self, server_url, ssl_options={}):
        """
        参数
        ----------
        server_url : str
            Triton 服务器的 gRPC 端点 URL。
        ssl_options : dict
            gRPC Python 客户端的 SSL 选项字典。
        """

        ssl = False
        root_certificates = None
        private_key = None
        certificate_chain = None

        if "ssl-grpc-use-ssl" in ssl_options:
            ssl = ssl_options["ssl-grpc-use-ssl"].lower() == "true"
        if "ssl-grpc-root-certifications-file" in ssl_options:
            root_certificates = ssl_options["ssl-grpc-root-certifications-file"]
        if "ssl-grpc-private-key-file" in ssl_options:
            private_key = ssl_options["ssl-grpc-private-key-file"]
        if "ssl-grpc-certificate-chain-file" in ssl_options:
            certificate_chain = ssl_options["ssl-grpc-certificate-chain-file"]

        channel_args = None
        if "localhost" in server_url:
            server_url = server_url.replace("localhost", "127.0.0.1")
            if ssl:
                channel_args = [("grpc.ssl_target_name_override", "localhost")]

        self._client = grpcclient.InferenceServerClient(
            url=server_url,
            ssl=ssl,
            root_certificates=root_certificates,
            private_key=private_key,
            certificate_chain=certificate_chain,
            channel_args=channel_args,
        )

    def get_model_config(self, model_name, num_retries):
        """
        获取模型配置。

        Parameters
        ----------
        model_name : str
            要获取配置的模型名称。
        num_retries : int
            等待模型加载的重试次数。

        Returns
        -------
        dict
            包含模型配置的字典。
        """

        self.wait_for_model_ready(model_name, num_retries)
        model_config_dict = self._client.get_model_config(model_name, as_json=True)
        return model_config_dict["config"]

    def get_model_repository_index(self):
        """
        返回包含模型仓库索引的 JSON 字典。
        """
        return self._client.get_model_repository_index(as_json=True)["models"]

    def is_model_ready(self, model_name: str) -> bool:
        """
        返回模型是否已加载到服务器上。
        """
        return self._client.is_model_ready(model_name)