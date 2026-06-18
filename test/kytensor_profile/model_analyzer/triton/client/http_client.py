#!/usr/bin/env python3

import logging

import gevent.ssl
import tritonclient.http as httpclient

from .client import TritonClient


class TritonHTTPClient(TritonClient):
    """
    TritonClient 的 HTTP 具体实现。
    """

    def __init__(self, server_url, ssl_options={}):
        """
        参数
        ----------
        server_url : str
            Triton 服务器的 HTTP 端点 URL。
        ssl_options : dict
            HTTP Python 客户端的 SSL 选项字典。
        """

        ssl = False
        client_ssl_options = {}
        ssl_context_factory = gevent.ssl._create_unverified_context
        insecure = True
        verify_peer = 0
        verify_host = 0

        if server_url.startswith("http://"):
            server_url = server_url.replace("http://", "", 1)
        elif server_url.startswith("https://"):
            ssl = True
            server_url = server_url.replace("https://", "", 1)
        if "ssl-https-ca-certificates-file" in ssl_options:
            client_ssl_options["ca_certs"] = ssl_options[
                "ssl-https-ca-certificates-file"
            ]
        if "ssl-https-client-certificate-file" in ssl_options:
            if (
                "ssl-https-client-certificate-type" in ssl_options
                and ssl_options["ssl-https-client-certificate-type"] == "PEM"
            ):
                client_ssl_options["certfile"] = ssl_options[
                    "ssl-https-client-certificate-file"
                ]
            else:
                logging.warning(
                    "使用 SSL 的 model-analyzer 必须传入 PEM 格式的客户端证书文件。"
                )
        if "ssl-https-private-key-file" in ssl_options:
            if (
                "ssl-https-private-key-type" in ssl_options
                and ssl_options["ssl-https-private-key-type"] == "PEM"
            ):
                client_ssl_options["keyfile"] = ssl_options[
                    "ssl-https-private-key-file"
                ]
            else:
                logging.warning(
                    "使用 SSL 的 model-analyzer 必须传入 PEM 格式的私钥文件。"
                )
        if "ssl-https-verify-peer" in ssl_options:
            verify_peer = ssl_options["ssl-https-verify-peer"]
        if "ssl-https-verify-host" in ssl_options:
            verify_host = ssl_options["ssl-https-verify-host"]
        if verify_peer != 0 and verify_host != 0:
            ssl_context_factory = None
            insecure = False

        self._client = httpclient.InferenceServerClient(
            url=server_url,
            ssl=ssl,
            ssl_options=client_ssl_options,
            ssl_context_factory=ssl_context_factory,
            insecure=insecure,
        )

    def get_model_repository_index(self):
        """
        返回包含模型仓库索引的字典。
        """
        return self._client.get_model_repository_index()

    def get_model_config(self, model_name, num_retries):
        """
        返回指定模型的配置。
        """
        self.wait_for_model_ready(model_name, num_retries)
        model_config_dict = self._client.get_model_config(model_name, as_json=True)
        return model_config_dict["config"]

    def is_model_ready(self, model_name: str) -> bool:
        """
        返回模型是否已加载到服务器上。
        """
        return self._client.is_model_ready(model_name)