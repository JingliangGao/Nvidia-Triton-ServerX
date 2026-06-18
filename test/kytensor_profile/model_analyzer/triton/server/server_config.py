#!/usr/bin/env python3

try:
    from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException
except ImportError:
    from model_analyzer_exceptions import TritonModelAnalyzerException


class TritonServerConfig:
    """
    用于设置 Triton Inference Server 参数的配置类。
    设置为 None 的参数将使用服务器默认值。
    """

    server_arg_keys = [
        # 模型仓库
        "model-repository",
        "model-control-mode",
        "strict-model-config",
        # HTTP
        "allow-http",
        "http-address",
        "http-port",
        # GRPC
        "allow-grpc",
        "grpc-address",
        "grpc-port",
        # 指标
        "allow-metrics",
    ]

    def __init__(self):
        """
        构造 TritonServerConfig。
        """

        self._server_args = {k: None for k in self.server_arg_keys}

    @classmethod
    def allowed_keys(cls):
        """
        Returns
        -------
        list of str
            可用于配置 tritonserver 实例的键列表。
        """

        snake_cased_keys = [key.replace("-", "_") for key in cls.server_arg_keys]
        return cls.server_arg_keys + snake_cased_keys

    def update_config(self, params=None):
        """
        允许从参数字典设置值。

        Parameters
        ----------
        params: dict
            键为 perf_analyzer 的允许参数。
        """

        if params:
            for key in params:
                self[key.strip().replace("_", "-")] = params[key]

    def to_cli_string(self):
        """
        将配置转换为服务器命令行参数字符串的实用函数。

        Returns
        -------
        str
            由所有已设置参数组成的命令字符串。
            例如: '--model-repository=/models --log-verbose=True'
        """

        return " ".join(
            [f"--{key}={val}" for key, val in self._server_args.items() if val]
        )

    def to_args_list(self):
        """
        将命令行字符串转换为参数列表的实用函数，
        同时考虑"智能"分隔符。注意在下面的示例中，
        只有第一个等号用作分隔符。

        Returns
        -------
        list
            由所有已设置参数组成的参数列表。
        """
        args_list = []
        args = self.to_cli_string().split()
        for arg in args:
            args_list += arg.split("=", 1)
        return args_list

    def copy(self):
        """
        Returns
        -------
        TritonServerConfig
            具有与此对象相同参数的对象。
        """

        config_copy = TritonServerConfig()
        config_copy.update_config(params=self._server_args)
        return config_copy

    def server_args(self):
        """
        Returns
        -------
        dict
            键为服务器参数名，值为其对应的值。
        """

        return self._server_args

    def __getitem__(self, key):
        """
        获取配置中参数的值。

        Parameters
        ----------
        key : str
            tritonserver 参数名。

        Returns
        -------
            此配置中该参数的值。
        """

        return self._server_args[key.strip().replace("_", "-")]

    def __setitem__(self, key, value):
        """
        设置配置中参数的值，在检查是否定义/支持后进行设置。

        Parameters
        ----------
        key : str
            tritonserver 参数名。
        value : (any)
            要设置给该参数的值。

        Raises
        ------
        TritonModelAnalyzerException
            如果该参数不受支持或在此配置类中未定义。
        """

        kebab_cased_key = key.strip().replace("_", "-")
        if kebab_cased_key in self._server_args:
            self._server_args[kebab_cased_key] = value
        else:
            raise TritonModelAnalyzerException(
                f"参数 '{key}' 不受 Triton Inference Server "
                "的 model analyzer 支持。"
            )