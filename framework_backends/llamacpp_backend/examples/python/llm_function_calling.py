'''
Copyright 2024. All Rights Reserved.
Author: Arthur Bin
Date: 2024-10-08 20:07:17
Description: file information
'''
import requests
import json
import signal

import asyncio
import argparse
import sys
import queue
from enum import Enum
from functools import partial
import numpy as np
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
from tritonclient.utils import *

def np_to_triton_dtype(np_dtype):
    if np_dtype == bool:
        return "BOOL"
    elif np_dtype == np.int8:
        return "INT8"
    elif np_dtype == np.int16:
        return "INT16"
    elif np_dtype == np.int32:
        return "INT32"
    elif np_dtype == np.int64:
        return "INT64"
    elif np_dtype == np.uint8:
        return "UINT8"
    elif np_dtype == np.uint16:
        return "UINT16"
    elif np_dtype == np.uint32:
        return "UINT32"
    elif np_dtype == np.uint64:
        return "UINT64"
    elif np_dtype == np.float16:
        return "FP16"
    elif np_dtype == np.float32:
        return "FP32"
    elif np_dtype == np.float64:
        return "FP64"
    elif np_dtype == np.object_ or np_dtype.type == np.bytes_:
        return "BYTES"
    return None

def prepare_tensor(input, name, shape, d_type):

    if isinstance(input, (list, np.ndarray)):
        data = np.array(input, dtype=d_type)
    else:
        data = np.array([input], dtype=d_type)
    t = grpcclient.InferInput(name, shape,
                        np_to_triton_dtype(d_type))
    t.set_data_from_numpy(data)
    return t

class RequestType(Enum):
    COMPLETIONS = 0
    OPENAI_COMPLETIONS = 1
    OPENAI_CHAT = 2
    TOKENIZE = 3
    DETOKENIZE = 4
    #LORA_SET = 5
    LORA_GET = 6
    # 取消任务
    TASK_CANCEL = 7
    # 所有任务终止
    TASK_ABORT = 8
    # 获取metrics
    METRICS = 9
    # 保存worker状态
    WORKER_SAVE = 10
    # 恢复worker状态
    WORKER_RESTORE = 11
    # 删除worker相关信息
    WORKER_ERASE = 12

def prepare_inputs_openai(request_type, request_id , text_input):
    inputs = [
        prepare_tensor(request_type, "request_type", [1], np.int32),
        prepare_tensor(request_id, "request_id", [1], np.object_),
        prepare_tensor(text_input, "text_input", [1], np.object_),
    ]

    return inputs

class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()

def callback(user_data, result, error):
    if error:
        print("error:", error)
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)

tool_json_str = '''{
    "model": "qwen2.5-3b",
    "tools": [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的时间，格式为YYYY-MM-DD HH:MM:SS",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
    ],
    "messages": [
        {"role": "user", "content": "请告诉我现在的时间。"}
    ]
}'''

json_str = '''{
  "temperature":0.7,
  "top_k":40,
  "top_p":0.95,
  "n_keep":0,
  "n_predict":-1,
  "cache_prompt":true,
  "stop":"<|im_end|>",
  "stream":true,
  "repeat_penalty":1.1
}'''

N_Keep = 0

def test_function_calling(client, name, user_data):

    inputs = prepare_inputs_openai(RequestType.OPENAI_CHAT.value, "chat-123", tool_json_str)

    client.async_stream_infer(name, inputs)

    answer = ""
    recv_count = 0
    final_response = False
    while final_response == False:
        data_item = user_data._completed_requests.get()
        final_response = data_item.get_response().parameters['triton_final_response'].bool_param
        if type(data_item) == InferenceServerException:
            print(data_item)
            sys.exit(1)
        else:
            output_data = data_item.as_numpy("text_output")
            if output_data is not None:
                out_str = output_data[0][0]
                print(out_str.decode('utf-8'), end='', flush=True)
                answer += out_str.decode('utf-8')

            output_token_data = data_item.as_numpy("token_output")
            if output_token_data is not None:
                print("output_token_data: ", output_token_data)
        recv_count += 1

    print()

# 声明全局变量
triton_client = None
model_name = "llamacpp"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:8001",
        help="Inference server URL and its gRPC port. Default is localhost:8001.",
    )
    parser.add_argument(
        "-s",
        "--streaming-mode",
        action="store_true",
        required=False,
        default=False,
        help="Enable streaming mode",
    )
    FLAGS = parser.parse_args()

    user_data = UserData()
    with grpcclient.InferenceServerClient(
        url=FLAGS.url,
        verbose=FLAGS.verbose,
    ) as triton_client:
        try:
            triton_client.start_stream(callback=partial(callback, user_data))
            test_function_calling(triton_client, model_name, user_data)

        except Exception as error:
            print(f"Encountered an error during request creation: {error}")
