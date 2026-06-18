#!/usr/bin/env python3

try:
    from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException
except ImportError:
    from model_analyzer_exceptions import TritonModelAnalyzerException

from .output_writer import OutputWriter


class FileWriter(OutputWriter):
    """
    将输出写入文件或标准输出。
    """

    def __init__(self, filename=None):
        """
        参数
        ----------
        filename : str
            要写入的文件或流的完整路径。
            如果 filename 为 None，则写入标准输出。
        """

        self._filename = filename

    def write(self, out, append=False):
        """
        将输出写入文件或标准输出。

        Parameters
        ----------
        out : str
            要写入文件或标准输出的字符串。

        Raises
        ------
        TritonModelAnalyzerException
            如果写入输出时发生错误或异常。
        """

        write_mode = "a+" if append else "w+"
        if self._filename:
            try:
                with open(self._filename, write_mode) as f:
                    f.write(out)
            except OSError as e:
                raise TritonModelAnalyzerException(e)
        else:
            print(out, end="")