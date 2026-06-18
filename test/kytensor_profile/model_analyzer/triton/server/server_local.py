#!/usr/bin/env python3

import logging
import os
import tempfile
from io import TextIOWrapper
from subprocess import DEVNULL, STDOUT, Popen, TimeoutExpired

import psutil

try:
    from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException
except ImportError:
    from model_analyzer_exceptions import TritonModelAnalyzerException

from .server import TritonServer

logger = logging.getLogger('model_analyzer.constants')


def find_kytensor_process():
    """
    查找 kytensor 进程的 psutil.Process 对象。

    Returns
    -------
    psutil.Process or None
        如果找到 kytensor 进程，则返回进程对象，否则返回 None。
    """
    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'] == 'kytensor':
            return proc
    return None


class TritonServerLocal(TritonServer):
    """
    TritonServer 接口的具体实现，以子进程方式在本地运行 tritonserver。
    """

    def __init__(self, path, config, gpus, log_path):
        """
        参数
        ----------
        path  : str
            tritonserver 可执行文件的绝对路径。
        config : TritonServerConfig
            包含此服务器实例参数的配置对象。
        gpus: list of str
            对 Triton 可见的 GPU UUID 列表。
        log_path: str
            Triton 日志文件的绝对路径。
        """

        self._tritonserver_process = None
        self._server_config = config
        self._server_path = path
        self._gpus = gpus
        self._log_path = log_path
        self._log_file = DEVNULL
        self._is_first_time_starting_server = True

        assert self._server_config[
            "model-repository"
        ], "Triton Server 需要设置 --model-repository 参数。"

    def start(self, env=None):
        """
        以子进程方式启动 tritonserver。
        """

        if self._server_path:
            # 创建命令列表并运行子进程
            cmd = [self._server_path]
            cmd += self._server_config.to_args_list()

            # 设置环境变量，使用用户配置的环境变量更新
            triton_env = os.environ.copy()

            if env:
                # 过滤需要使用环境查找的变量
                for variable, value in env.items():
                    if value.find("$") == -1:
                        triton_env[variable] = value
                    else:
                        # 收集需要查找的变量，交给 shell 处理
                        triton_env[variable] = os.path.expandvars(value)

            if self._log_path:
                try:
                    if self._is_first_time_starting_server:
                        if os.path.exists(self._log_path):
                            os.remove(self._log_path)
                    self._log_file = open(self._log_path, "a+")
                except OSError as e:
                    raise TritonModelAnalyzerException(e)
            else:
                self._log_file = tempfile.NamedTemporaryFile()

            self._is_first_time_starting_server = False

            # 构造 Popen 命令
            try:
                self._tritonserver_process = Popen(
                    cmd,
                    stdout=self._log_file,
                    stderr=STDOUT,
                    start_new_session=True,
                    universal_newlines=True,
                    env=triton_env,
                )

                logger.debug("Triton Server 已启动。")
            except Exception as e:
                raise TritonModelAnalyzerException(e)

    def stop(self):
        """
        停止正在运行的 tritonserver。
        """

        # 终止进程，捕获输出
        if self._tritonserver_process is not None:
            self._tritonserver_process.terminate()
            try:
                self._tritonserver_process.communicate(
                    timeout=5
                )
            except TimeoutExpired:
                self._tritonserver_process.kill()
                self._tritonserver_process.communicate()
            self._tritonserver_process = None
            if self._log_path:
                self._log_file.close()
            logger.debug("Triton Server 已停止。")

    def cpu_stats(self, server_process=None):
        """
        返回 CPU 内存使用量和系统可用内存（MB）。

        Parameters
        ----------
        server_process : psutil.Process, optional
            要监控的进程对象。如果未提供，
            使用内部的 tritonserver 进程。

        Returns
        -------
        tuple
            包含 (process_uss_mb, system_available_mb) 的元组。
        """
        if server_process is None:
            server_process = self._get_internal_process()

        if server_process:
            try:
                process_memory_info = server_process.memory_full_info()
                process_uss_mb = process_memory_info.uss // 1.0e6
            except psutil.AccessDenied:
                # 获取 smaps_rollup 需要 root 权限
                # 回退到使用 memory_info() 获取 RSS
                try:
                    process_memory_info = server_process.memory_info()
                    process_uss_mb = process_memory_info.rss // 1.0e6
                except psutil.AccessDenied:
                    process_uss_mb = 0.0

            system_memory_info = psutil.virtual_memory()
            return process_uss_mb, (system_memory_info.available // 1.0e6)
        else:
            return 0.0, 0.0

    def cpu_percent(self, interval=1.0, server_process=None):
        """
        返回 tritonserver 进程和系统的 CPU 使用率百分比。

        Parameters
        ----------
        interval : float
            测量 CPU 使用率的时间间隔（秒）。
            值为 0.0 时返回瞬时 CPU 使用率（首次调用可能不准确）。
        server_process : psutil.Process, optional
            要监控的进程对象。如果未提供，
            使用内部的 tritonserver 进程。

        Returns
        -------
        tuple
            包含 (process_cpu_percent, system_cpu_percent) 的元组。
            - process_cpu_percent: tritonserver 进程的 CPU 使用率
            - system_cpu_percent: 系统整体 CPU 使用率
        """
        if server_process is None:
            server_process = self._get_internal_process()

        if server_process:
            process_cpu_percent = server_process.cpu_percent(interval=interval)
            system_cpu_percent = psutil.cpu_percent(interval=0)
            return process_cpu_percent, system_cpu_percent
        else:
            return 0.0, 0.0

    def _get_internal_process(self):
        """
        获取内部 tritonserver 进程的 psutil.Process 对象。

        Returns
        -------
        psutil.Process or None
            如果找到进程对象则返回，否则返回 None。
        """
        if self._tritonserver_process:
            try:
                return psutil.Process(self._tritonserver_process.pid)
            except psutil.NoSuchProcess:
                return None
        return None

    def log_file(self) -> TextIOWrapper:
        """返回日志文件对象。"""
        return self._log_file