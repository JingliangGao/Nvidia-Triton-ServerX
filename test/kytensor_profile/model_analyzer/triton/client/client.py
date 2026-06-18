#!/usr/bin/env python3

import logging
import time
from subprocess import DEVNULL

try:
    from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException
except ImportError:
    from model_analyzer_exceptions import TritonModelAnalyzerException

logger = logging.getLogger('model_analyzer.constants')


class TritonClient:
    """
    定义由 TritonClientFactory 创建的对象的接口基类。
    """

    def wait_for_server_ready(
        self,
        num_retries,
        sleep_time=1,
        log_file=None,
    ):
        """
        Parameters
        ----------
        num_retries : int
            在向服务器发送就绪状态请求前的重试次数。
        sleep_time: int
            重试之间的等待时间（秒）。
        log_file: TextIOWrapper
            服务器输出日志文件。

        Raises
        ------
        TritonModelAnalyzerException
            如果在指定重试次数内无法确定服务器就绪状态。
        """

        retries = num_retries
        while retries > 0:
            try:
                if self._client.is_server_ready():
                    time.sleep(sleep_time)
                    return
                else:
                    self._check_for_triton_log_errors(log_file)
                    time.sleep(sleep_time)
                    retries -= 1
            except Exception as e:
                # 记录连接失败的详细信息以便调试
                if retries == num_retries or retries % 10 == 0:
                    logger.debug(
                        f"连接 Triton 服务器失败（第 {num_retries - retries + 1}/{num_retries} 次尝试）: {e}"
                    )
                self._check_for_triton_log_errors(log_file)
                time.sleep(sleep_time)
                retries -= 1
                if retries == 0:
                    raise TritonModelAnalyzerException(e)
        raise TritonModelAnalyzerException(
            "无法确定服务器就绪状态，超过重试次数。"
        )

    def load_model(self, model_name, variant_name="", config_str=None):
        """
        在显式模型控制模式下，请求推理服务器加载特定模型。

        Parameters
        ----------
        model_name : str
            模型名称。
        variant_name: str
            模型变体名称。
        config_str: str
            用于加载模型的可选配置字符串。

        Returns
        ------
        int or None
            失败时返回 -1，成功返回 None。
        """

        variant_name = variant_name if variant_name else model_name

        try:
            self._client.load_model(model_name, config=config_str)
            logger.debug(f"模型 {variant_name} 已加载")
            return None
        except Exception as e:
            logger.info(f"模型 {variant_name} 加载失败: {e}")
            if "polling is enabled" in e.message():
                raise TritonModelAnalyzerException(
                    "远程 Tritonserver 需要在 EXPLICIT 模式下启动"
                )
            return -1

    def unload_model(self, model_name):
        """
        在显式模型控制模式下，请求推理服务器卸载特定模型。

        Parameters
        ----------
        model_name : str
            要从仓库中加载的模型名称。

        Raises
        ------
        TritonModelAnalyzerException
            如果服务器抛出异常。

        Returns
        ------
        int or None
            失败时返回 -1，成功返回 None。
        """

        try:
            self._client.unload_model(model_name)
            logger.debug(f"模型 {model_name} 已卸载")
            return None
        except Exception as e:
            logger.info(f"模型 {model_name} 卸载失败: {e}")
            return -1

    def wait_for_model_ready(self, model_name, num_retries, sleep_time=1):
        """
        等待模型准备就绪。

        Parameters
        ----------
        model_name : str
            要从仓库中加载的模型名称。
        num_retries : int
            在向服务器发送就绪状态请求前的重试次数。

        Raises
        ------
        TritonModelAnalyzerException
            如果在指定重试次数内无法确定模型就绪状态。

        Returns
        ------
        int or None
            失败时返回 -1，成功返回 None。
        """

        retries = num_retries
        error = None
        while retries > 0:
            try:
                if self._client.is_model_ready(model_name):
                    return None
                else:
                    time.sleep(sleep_time)
                    retries -= 1
            except Exception as e:
                error = e
                time.sleep(sleep_time)
                retries -= 1

        logger.info(f"模型 {model_name} 就绪检查失败: {error}")
        return -1

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
        dict or None
            包含模型配置的字典。
        """

        self.wait_for_model_ready(model_name, num_retries)
        model_config_dict = self._client.get_model_config(model_name)
        return model_config_dict

    def is_server_ready(self):
        """
        返回服务器是否就绪。

        Returns
        -------
        bool
            服务器就绪返回 True，否则返回 False。
        """
        return self._client.is_server_ready()

    def infer(self, model_name, inputs, outputs=None, request_id=None):
        """
        使用底层 Triton 客户端执行推理请求。

        Parameters
        ----------
        model_name : str
            要推理的模型名称。
        inputs : list
            Triton InferInput 对象列表。
        outputs : list, optional
            Triton InferRequestedOutput 对象列表。
        request_id : str, optional
            请求标识符。
        """
        return self._client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
            request_id=request_id,
        )

    def _check_for_triton_log_errors(self, log_file):
        """检查 Triton 日志中是否有错误。"""
        if not log_file or log_file == DEVNULL:
            return

        log_file.seek(0)
        log_output = log_file.read()

        if not type(log_output) == str:
            log_output = log_output.decode("utf-8")

        if log_output:
            if "Unexpected argument:" in log_output:
                error_start = log_output.find("Unexpected argument:")
                raise TritonModelAnalyzerException(
                    f"错误: TritonServer 未能成功启动\n\n{log_output[error_start:]}"
                )