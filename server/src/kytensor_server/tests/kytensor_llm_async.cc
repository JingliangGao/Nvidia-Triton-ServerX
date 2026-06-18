// Copyright 2025. All Rights Reserved.
// Author: Arthur Bin
// Date: 2025-10-16
// Description: kytensor client test for llamacpp

#include <unistd.h>
#include <signal.h>

#include <condition_variable>
#include <iostream>
#include <memory>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include "../src/kytensor_client.h"

namespace kc = kytensor::client;

using ResultList = std::vector<std::shared_ptr<kc::InferResult>>;

// Enum for request types
enum class RequestType {
	COMPLETIONS = 0,
	TOKENIZE = 1,
	DETOKENIZE = 2,
	//LORA_SET = 3,
	LORA_GET = 4,
	TASK_CANCEL = 5, //取消任务
	TASK_ABORT = 6, //所有任务终止
	METRICS = 7, //获取metrics
	WORKER_SAVE = 8,  //保存worker状态
	WORKER_RESTORE = 9, //恢复worker状态
	WORKER_ERASE = 10, //删除worker相关信息
};

// Global variables for synchronization and control
std::mutex mutex_;
std::condition_variable cv_;
std::queue<std::shared_ptr<kc::InferResult>> completed_requests_queue;
bool should_exit = false;

#define FAIL_IF_ERR(X, MSG)                                        \
{                                                                	\
	kc::Error err = (X);                                           	\
	if (!err.IsOk()) {                                             	\
		std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
		exit(1);                                                     \
	}                                                              	  \
}

void Usage(char** argv, const std::string& msg = std::string()) {
	if (!msg.empty()) {
		std::cerr << "error: " << msg << std::endl;
	}

	std::cerr << "Usage: " << argv[0] << " [options]" << std::endl;
	std::cerr << "\t-v" << std::endl;
	std::cerr << "\t-u <URL for inference service and its gRPC port>" << std::endl;
	std::cerr << "\t-s Enable streaming mode" << std::endl;

	exit(1);
}

void callback(kc::InferResult* result) {
	std::shared_ptr<kc::InferResult> result_ptr(result);
	{
		std::lock_guard<std::mutex> lk(mutex_);
		completed_requests_queue.push(result_ptr);
	}
	cv_.notify_all();
}

// Prepare input tensors for inference
void prepare_inputs(
    const std::unique_ptr<kc::KyTensorClient>& client,
    const std::string& model_name,
    RequestType request_type, const std::string& request_id,
    const std::string& text_input, float temperature,
    int32_t top_k, float top_p, int32_t n_keep,
    int32_t n_predict, bool cache_prompt,
    const std::string& stop, bool stream,
    const std::vector<float>& lora_scale,
    float repeat_penalty) {

	std::vector<kc::InferInput*> inputs;
	kc::InferInput* input;

	// Convert enum to int32_t for the request type
	int32_t request_type_int = static_cast<int32_t>(request_type);

	// request_type
	FAIL_IF_ERR(kc::InferInput::Create(&input, "request_type", {1}, "INT32"), "unable to create 'request_type'");
	std::shared_ptr<kc::InferInput> request_type_input(input);
	FAIL_IF_ERR(request_type_input->Reset(), "unable to reset 'request_type'");
	FAIL_IF_ERR(request_type_input->AppendRaw(reinterpret_cast<const uint8_t*>(&request_type_int), sizeof(int32_t)), "unable to set data for 'request_type'");
	inputs.push_back(request_type_input.get());

	// request_id
	FAIL_IF_ERR(kc::InferInput::Create(&input, "request_id", {1}, "BYTES"), "unable to create 'request_id'");
	std::shared_ptr<kc::InferInput> request_id_input(input);
	FAIL_IF_ERR(request_id_input->Reset(), "unable to reset 'request_id'");
	request_id_input->SetBinaryData(true);
	FAIL_IF_ERR(request_id_input->AppendFromString({request_id}), "unable to set data for 'request_id'");
	inputs.push_back(request_id_input.get());

	// text_input
	FAIL_IF_ERR(kc::InferInput::Create(&input, "text_input", {1}, "BYTES"), "unable to create 'text_input'");
	std::shared_ptr<kc::InferInput> text_input_input(input);
	FAIL_IF_ERR(text_input_input->Reset(), "unable to reset 'text_input'");
	text_input_input->SetBinaryData(true);
	FAIL_IF_ERR(text_input_input->AppendFromString({text_input}), "unable to set data for 'text_input'");
	inputs.push_back(text_input_input.get());

	// temperature_input
	FAIL_IF_ERR(kc::InferInput::Create(&input, "temperature_input", {1}, "FP32"), "unable to create 'temperature_input'");
	std::shared_ptr<kc::InferInput> temperature_input(input);
	FAIL_IF_ERR(temperature_input->Reset(), "unable to reset 'temperature_input'");
	FAIL_IF_ERR(temperature_input->AppendRaw(reinterpret_cast<const uint8_t*>(&temperature), sizeof(float)), "unable to set data for 'temperature_input'");
	inputs.push_back(temperature_input.get());

	// top_k_input
	FAIL_IF_ERR(kc::InferInput::Create(&input, "top_k_input", {1}, "INT32"), "unable to create 'top_k_input'");
	std::shared_ptr<kc::InferInput> top_k_input(input);
	FAIL_IF_ERR(top_k_input->Reset(), "unable to reset 'top_k_input'");
	FAIL_IF_ERR(top_k_input->AppendRaw(reinterpret_cast<const uint8_t*>(&top_k), sizeof(int32_t)), "unable to set data for 'top_k_input'");
	inputs.push_back(top_k_input.get());

	// top_p_input
	FAIL_IF_ERR(kc::InferInput::Create(&input, "top_p_input", {1}, "FP32"), "unable to create 'top_p_input'");
	std::shared_ptr<kc::InferInput> top_p_input(input);
	FAIL_IF_ERR(top_p_input->Reset(), "unable to reset 'top_p_input'");
	FAIL_IF_ERR(top_p_input->AppendRaw(reinterpret_cast<const uint8_t*>(&top_p), sizeof(float)), "unable to set data for 'top_p_input'");
	inputs.push_back(top_p_input.get());

	// n_keep_input
	FAIL_IF_ERR(kc::InferInput::Create(&input, "n_keep_input", {1}, "INT32"), "unable to create 'n_keep_input'");
	std::shared_ptr<kc::InferInput> n_keep_input(input);
	FAIL_IF_ERR(n_keep_input->Reset(), "unable to reset 'n_keep_input'");
	FAIL_IF_ERR(n_keep_input->AppendRaw(reinterpret_cast<const uint8_t*>(&n_keep), sizeof(int32_t)), "unable to set data for 'n_keep_input'");
	inputs.push_back(n_keep_input.get());

	// n_predict_input
	FAIL_IF_ERR(kc::InferInput::Create(&input, "n_predict_input", {1}, "INT32"), "unable to create 'n_predict_input'");
	std::shared_ptr<kc::InferInput> n_predict_input(input);
	FAIL_IF_ERR(n_predict_input->Reset(), "unable to reset 'n_predict_input'");
	FAIL_IF_ERR(n_predict_input->AppendRaw(reinterpret_cast<const uint8_t*>(&n_predict), sizeof(int32_t)), "unable to set data for 'n_predict_input'");
	inputs.push_back(n_predict_input.get());

	// cache_prompt_input
	FAIL_IF_ERR(kc::InferInput::Create(&input, "cache_prompt_input", {1}, "BOOL"), "unable to create 'cache_prompt_input'");
	std::shared_ptr<kc::InferInput> cache_prompt_input(input);
	FAIL_IF_ERR(cache_prompt_input->Reset(), "unable to reset 'cache_prompt_input'");
	FAIL_IF_ERR(cache_prompt_input->AppendRaw(reinterpret_cast<const uint8_t*>(&cache_prompt), sizeof(bool)), "unable to set data for 'cache_prompt_input'");
	inputs.push_back(cache_prompt_input.get());

	// stop_input
	FAIL_IF_ERR(kc::InferInput::Create(&input, "stop_input", {1}, "BYTES"), "unable to create 'stop_input'");
	std::shared_ptr<kc::InferInput> stop_input(input);
	FAIL_IF_ERR(stop_input->Reset(), "unable to reset 'stop_input'");
	stop_input->SetBinaryData(true);
	FAIL_IF_ERR(stop_input->AppendFromString({stop}), "unable to set data for 'stop_input'");
	inputs.push_back(stop_input.get());

	// stream_input
	FAIL_IF_ERR(kc::InferInput::Create(&input, "stream_input", {1}, "BOOL"), "unable to create 'stream_input'");
	std::shared_ptr<kc::InferInput> stream_input(input);
	FAIL_IF_ERR(stream_input->Reset(), "unable to reset 'stream_input'");
	FAIL_IF_ERR(stream_input->AppendRaw(reinterpret_cast<const uint8_t*>(&stream), sizeof(bool)), "unable to set data for 'stream_input'");
	inputs.push_back(stream_input.get());

	// lora_scale_input
	FAIL_IF_ERR(kc::InferInput::Create(&input, "lora_scale_input", {static_cast<int64_t>(lora_scale.size())}, "FP32"), "unable to create 'lora_scale_input'");
	std::shared_ptr<kc::InferInput> lora_scale_input(input);
	FAIL_IF_ERR(lora_scale_input->Reset(), "unable to reset 'lora_scale_input'");
	if (!lora_scale.empty()) {
		FAIL_IF_ERR(lora_scale_input->AppendRaw(reinterpret_cast<const uint8_t*>(lora_scale.data()), lora_scale.size() * sizeof(float)), "unable to set data for 'lora_scale_input'");
	}
	inputs.push_back(lora_scale_input.get());

	// repeat_penalty_input
	FAIL_IF_ERR(kc::InferInput::Create(&input, "repeat_penalty_input", {1}, "FP32"), "unable to create 'repeat_penalty_input'");
	std::shared_ptr<kc::InferInput> repeat_penalty_input(input);
	FAIL_IF_ERR(repeat_penalty_input->Reset(), "unable to reset 'repeat_penalty_input'");
	FAIL_IF_ERR(repeat_penalty_input->AppendRaw(reinterpret_cast<const uint8_t*>(&repeat_penalty), sizeof(float)), "unable to set data for 'repeat_penalty_input'");
	inputs.push_back(repeat_penalty_input.get());

	kc::InferRequestedOutput* text_output;
	FAIL_IF_ERR(
		kc::InferRequestedOutput::Create(&text_output, "text_output"),
		"unable to get text_output");
	std::shared_ptr<kc::InferRequestedOutput> text_output_ptr;
	text_output_ptr.reset(text_output);
	std::vector<const kc::InferRequestedOutput*> outputs = { text_output_ptr.get() };

	kc::InferOptions options(model_name);
	size_t repeat_cnt = 1;
	size_t done_cnt = 0;
	for (size_t i = 0; i < repeat_cnt; i++) {
		FAIL_IF_ERR(client->AsyncInfer(
			[&, i](kc::InferResult* result) {
				{
					std::shared_ptr<kc::InferResult> result_ptr;
					result_ptr.reset(result);
					std::lock_guard<std::mutex> lk(mutex_);
					std::cout << "Callback no." << i << " is called " << std::endl;
					done_cnt++;
					if (result_ptr->RequestStatus().IsOk()) {
						// Get text output if available
						// std::vector<std::string> output_strings;
						// err = result->StringData("text_output", &output_strings);
						// if (err.IsOk() && !output_strings.empty()) {
						// 	std::string out_str = output_strings[0];
						// 	std::cout << out_str << std::endl;
						// }
					} else {
						std::cerr << "error: Inference failed: "
								<< result_ptr->RequestStatus() << std::endl;
						exit(1);
					}
				}
				cv_.notify_all();
			},
			options, inputs, outputs), "unable to run tokenize model");
	}

	// Wait until all callbacks are invoked
	{
		std::unique_lock<std::mutex> lk(mutex_);
		cv_.wait(lk, [&]() {
			if (done_cnt >= repeat_cnt) {
				return true;
			} else {
				return false;
			}
		});
	}
	if (done_cnt == repeat_cnt) {
		std::cout << "All done" << std::endl;
	} else {
		std::cerr << "Done cnt: " << done_cnt
				<< " does not match repeat cnt: " << repeat_cnt << std::endl;
		exit(1);
	}

	// deferred response
	bool callback_invoked = false;
	std::shared_ptr<kc::InferResult> result_placeholder;
	FAIL_IF_ERR(
		client->AsyncInfer(
			[&](kc::InferResult* result) {
				{
					std::shared_ptr<kc::InferResult> result_ptr;
					result_ptr.reset(result);
					// Defer the response retrieval to main thread
					std::lock_guard<std::mutex> lk(mutex_);
					callback_invoked = true;
					result_placeholder = std::move(result_ptr);
				}
				cv_.notify_all();
			},
			options, inputs, outputs),
		"unable to run model");

	// Ensure callback is completed
	{
	  std::unique_lock<std::mutex> lk(mutex_);
	  cv_.wait(lk, [&]() { return callback_invoked; });
	}

	// Get deferred response
	std::cout << "Getting results from deferred response" << std::endl;
	if (result_placeholder->RequestStatus().IsOk()) {
		// Get text output if available
		std::vector<std::string> output_strings;
		kc::Error err = result_placeholder->StringData("text_output", &output_strings);
		if (err.IsOk() && !output_strings.empty()) {
			std::string out_str = output_strings[0];
			std::cout << out_str << std::endl;
		}
	} else {
	  std::cerr << "error: Inference failed: "
				<< result_placeholder->RequestStatus() << std::endl;
	  exit(1);
	}

}

// Chat completion function (request type 0)
void test_async_infer(
    const std::unique_ptr<kc::KyTensorClient>& client,
    const std::string& model_name,
    const std::string& prompt) {

	prepare_inputs(client, model_name, RequestType::COMPLETIONS, "chat-123", prompt,
		0.7f, 40, 0.95f, 0, 100, true, "<|im_end|>", false, std::vector<float>(), 1.1f);

}

int main(int argc, char** argv) {
	bool verbose = false;
	std::string url("localhost:8001");

  // Parse commandline...
  int opt;
	while ((opt = getopt(argc, argv, "vu:s")) != -1) {
		switch (opt) {
			case 'v':
				verbose = true;
				break;
			case 'u':
				url = optarg;
				break;
			case 's':
				// Streaming mode is always enabled in this version
				break;
			case '?':
				Usage(argv);
				break;
		}
	}

	std::cout << "Connecting to: " << url << std::endl;

	// Create a InferenceServerGrpcClient instance to communicate with the server using gRPC protocol.
	std::unique_ptr<kc::KyTensorClient> client;
	FAIL_IF_ERR(
		kc::KyTensorClient::Create(&client, verbose),
		"unable to create grpc client");

	std::string model_name = "llamacpp";

	// Process the chat completion
	test_async_infer(client, model_name, "你是谁");


	std::cout << "Client exiting..." << std::endl;

	return 0;
}