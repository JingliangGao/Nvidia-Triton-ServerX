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
    # 针对标量和数组分别处理
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

def prepare_inputs(request_type, request_id , text_input, temperature,
                   top_k, top_p, n_keep, n_predict,
                   cache_prompt, stop, stream, lora_scale, repeat_penalty=1.0):
    inputs = [
        prepare_tensor(request_type, "request_type", [1], np.int32),
        prepare_tensor(request_id, "request_id", [1], np.object_),
        prepare_tensor(text_input, "text_input", [1], np.object_),
        prepare_tensor(temperature, "temperature_input", [1], np.float32),
        prepare_tensor(top_k, "top_k_input", [1], np.int32),
        prepare_tensor(top_p, "top_p_input", [1], np.float32),
        prepare_tensor(n_keep, "n_keep_input", [1], np.int32),
        prepare_tensor(n_predict, "n_predict_input", [1], np.int32),
        prepare_tensor(cache_prompt, "cache_prompt_input", [1], bool),
        prepare_tensor(stop, "stop_input", [1], np.object_),
        prepare_tensor(stream, "stream_input", [1], bool),
        prepare_tensor(lora_scale, "lora_scale_input", [len(lora_scale)], np.float32),
        prepare_tensor(repeat_penalty, "repeat_penalty_input", [1], np.float32),
    ]

    return inputs

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

Chat_context = []

System_prompt = f"<|im_start|>system\n无<|im_end|>\n"

def trim(s):
    """
    删除字符串两端的空白字符
    """
    return s.strip()

def trim_trailing(s):
    """
    删除字符串末尾的空白字符
    """
    return s.rstrip()

def format_prompt_string():
    #formatted_chat = f"{INSTRUCTION}"
    formatted_chat = System_prompt

    for index, line in enumerate(Chat_context):
        if index % 2 == 0:
            formatted_chat += f"<|im_start|>user\n{line}<|im_end|>\n"
        else:
            formatted_chat += f"<|im_start|>assistant\n{line}<|im_end|>\n"

    return formatted_chat

json_str = '''{
  "temperature":0.7,
  "top_k":40,
  "top_p":0.95,
  "n_keep":0,
  "n_predict":-1,
  "cache_prompt":true,
  "stop":"<|im_end|>",
  "stream":true,
  "repeat_penalty":1.1,
  "lora":[{"id":0,"scale":0.0},
           {"id":1,"scale":0.0}
          ]
}'''

def format_openai_prompt_string():
    """
    将聊天上下文转换为 OpenAI JSON 格式，包含 messages 字段
    """
    messages = [
        {"role": "system", "content": ""}
    ]

    for index, line in enumerate(Chat_context):
        if index % 2 == 0:
            messages.append({"role": "user", "content": line})
        else:
            messages.append({"role": "assistant", "content": line})

    # 构造包含 messages 字段的 JSON 对象

    data = json.loads(json_str)
    data["messages"] = messages
    formatted_chat = json.dumps(data, ensure_ascii=False)
    return formatted_chat

def tokenize(client, name, user_data, text):
    inputs = prepare_inputs(RequestType.TOKENIZE.value, "chat-123", text, 0.7, 40, 0.95,
                                0, 1024, True, "\n### Human:", True, [0.0, 0.0])

    client.async_stream_infer(name, inputs)

    recv_count = 0
    final_response = False
    output_data = []
    while final_response == False:
        data_item = user_data._completed_requests.get()
        final_response = data_item.get_response().parameters['triton_final_response'].bool_param
        if type(data_item) == InferenceServerException:
            client.stop_stream()
            print(data_item)
            sys.exit(1)
        else:
            output_data = data_item.as_numpy("token_output")
            if output_data is not None:
               print("token_output: ", output_data)

        recv_count += 1

    return output_data

N_Keep = 0
def chat_completion(client, name, user_data, question):
    Chat_context.append(trim_trailing(question))
    prompt = trim_trailing(format_prompt_string())
    prompt += f"\n<|im_start|>assistant\n"
    n_keep = N_Keep

    # set input tensor
    # 每次推理前设置request_id, 每次请求都要有一个唯一的request_id，
    # 若不设置则无法使用停止功能
    inputs = prepare_inputs(RequestType.COMPLETIONS.value, "chat-123", prompt, 0.7, 40, 0.95,
                                n_keep, -1, True, "<|im_end|>", True, [], 1.1)

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
    Chat_context.append(trim(answer))
    if len(Chat_context) == 4:
        del Chat_context[:2]

def chat_openai_completion(client, name, user_data, question):
    Chat_context.append(trim_trailing(question))
    prompt = trim_trailing(format_openai_prompt_string())
    #print(prompt)

    # set input tensor
    # 每次推理前设置request_id, 每次请求都要有一个唯一的request_id，
    # 若不设置则无法使用停止功能
    inputs = prepare_inputs_openai(RequestType.OPENAI_CHAT.value, "chat-123", prompt)

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
                out_str_json = json.loads(out_str.decode('utf-8'))

                if isinstance(out_str_json, list):
                    s = out_str_json[0]['choices'][0]['delta']['content']
                    print(s, end='', flush=True)
                    answer += s

            output_token_data = data_item.as_numpy("token_output")
            if output_token_data is not None:
                print("output_token_data: ", output_token_data)
        recv_count += 1

    print()
    Chat_context.append(trim(answer))
    if len(Chat_context) == 4:
        del Chat_context[:2]

# 声明全局变量
triton_client = None
model_name = "llamacpp"

def handle_ctrl_c(sig, frame):

    #print("检测到 Ctrl+C，正在取消任务...")
    #inputs = prepare_inputs(RequestType.TASK_CANCEL.value, "chat-123", "", 0.0, 0, 0,
    #                            0, 0, False, "\n### Human:", True, [])

    # 确保推理服务中的llm只有一个任务在运行，多任务运行时终止存在问题，谨慎使用
    print("检测到 Ctrl+C，正在终止任务...")
    inputs = prepare_inputs(RequestType.TASK_ABORT.value, "chat-123", "", 0.0, 0, 0,
                                0, 0, False, "\n### Human:", True, [])

    triton_client.async_stream_infer(model_name, inputs)
    final_response = False
    while final_response == False:
        data_item = user_data._completed_requests.get()
        final_response = data_item.get_response().parameters['triton_final_response'].bool_param
        if type(data_item) == InferenceServerException:
            print(data_item)
            sys.exit(1)

    triton_client.stop_stream()
    sys.exit(1)
    #if triton_client is not None:
    #    triton_client.stop_stream(cancel_requests=True)

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
    parser.add_argument(
        "-o",
        "--openai-mode",
        action="store_true",
        required=False,
        default=False,
        help="Use OpenAI-compatible chat mode (calls chat_openai_completion). Default uses chat_completion.",
    )
    FLAGS = parser.parse_args()

    user_data = UserData()
    format_prompt_string()
    signal.signal(signal.SIGINT, handle_ctrl_c)
    with grpcclient.InferenceServerClient(
        url=FLAGS.url,
        verbose=FLAGS.verbose,
    ) as triton_client:
        try:
            triton_client.start_stream(callback=partial(callback, user_data))
            # 获取要保存系统提示词的token，并将要保留的token数量传给N_Keep
            token = tokenize(triton_client, model_name, user_data, System_prompt)
            arr = np.array(token)
            if arr.ndim == 2:
                N_Keep = arr.shape[1]
            else:
                N_Keep = arr.size
            print("N_Keep len: ", N_Keep)

            while True:
                question = input("> ")
                if question.lower() in ["exit", "quit"]:
                    triton_client.stop_stream()
                    break
                if FLAGS.openai_mode:
                    chat_openai_completion(triton_client, model_name, user_data, question)
                else:
                    chat_completion(triton_client, model_name, user_data, question)

        except Exception as error:
            print(f"Encountered an error during request creation: {error}")
