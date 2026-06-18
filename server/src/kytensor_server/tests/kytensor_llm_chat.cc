// Copyright 2024. All Rights Reserved.
// Author: Arthur Bin
// Date: 2024-10-08
// Description: C++ streaming chat client for llamacpp backend

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

// Chat context to maintain conversation history
std::vector<std::string> chat_context;

// System prompt
std::string system_prompt = "<|im_start|>system\n无<|im_end|>\n";

// Function to trim trailing whitespace
std::string trim_trailing(const std::string& s) {
	size_t end_pos = s.find_last_not_of(" \t\n\r\f\v");
	if (end_pos != std::string::npos) {
		return s.substr(0, end_pos + 1);
	}
	return s;
}

// Format the prompt string from chat context
std::string format_prompt_string() {
	std::string formatted_chat = system_prompt;

	for (size_t i = 0; i < chat_context.size(); ++i) {
		//std::cout << "chat_context " << i << ": " << chat_context[i] << std::endl;
		if (i % 2 == 0) {
			formatted_chat += "<|im_start|>user\n" + chat_context[i] + "<|im_end|>\n";
		} else {
			formatted_chat += "<|im_start|>assistant\n" + chat_context[i] + "<|im_end|>\n";
		}
	}

	return formatted_chat;
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

// Signal handler for Ctrl+C
void signal_handler(int signal) {
	std::cout << "\nDetected Ctrl+C, terminating task..." << std::endl;
	should_exit = true;
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
	FAIL_IF_ERR(client->AsyncStreamInfer(options, inputs, outputs), "unable to run tokenize model");

}

void process_result(RequestType request_type)
{
	// Process the streaming responses
	bool final_response = false;
	std::string answer = "";

	while (!final_response && !should_exit) {
		std::shared_ptr<kc::InferResult> result;
		{
			std::unique_lock<std::mutex> lk(mutex_);
			cv_.wait(lk, [&] { return !completed_requests_queue.empty() || should_exit; });
			if (should_exit) {
				break;
			}
			result = completed_requests_queue.front();
			completed_requests_queue.pop();
		}

		// Check if this is a final response
		bool is_final = false;
		kc::Error err = result->IsFinalResponse(&is_final);
		if (err.IsOk() && is_final) {
			final_response = true;
		}

		// Check request status
		kc::Error status = result->RequestStatus();
		if (!status.IsOk()) {
			std::cout << "Error: " << status.Message() << std::endl;
			continue;
		}

		// Get text output if available
		std::vector<std::string> output_strings;
		err = result->StringData("text_output", &output_strings);
		if (err.IsOk() && !output_strings.empty()) {
			std::string out_str = output_strings[0];
			std::cout << out_str;
			std::cout.flush();
			answer += out_str;
		}

		// Get token output if available
		const uint8_t* token_raw_data;
		size_t token_byte_size;
		err = result->RawData("token_output", &token_raw_data, &token_byte_size);
		if (err.IsOk() && token_raw_data != nullptr && token_byte_size > 0) {
			size_t num_tokens = token_byte_size / sizeof(int32_t);
			std::vector<int32_t> token_data(num_tokens);
			memcpy(token_data.data(), token_raw_data, token_byte_size);
			std::cout << "output_token_data: ";
			for (int32_t token : token_data) {
				std::cout << token << " ";
			}
			std::cout << std::endl;
		}
	}

	std::cout << std::endl;
	if (request_type == RequestType::COMPLETIONS)
		chat_context.push_back(trim_trailing(answer));
}

// Tokenize function
void tokenize(
    const std::unique_ptr<kc::KyTensorClient>& client,
    const std::string& model_name,
    const std::string& text) {

	prepare_inputs( client, model_name, RequestType::TOKENIZE, "chat-123",  text,
		0.7f, 40, 0.95f, 0, 1024, true, "\n### Human:", true, {0.0f, 0.0f},
		1.0f);

	process_result(RequestType::TOKENIZE);
}

// Chat completion function (request type 0)
void chat_completion(
    const std::unique_ptr<kc::KyTensorClient>& client,
    const std::string& model_name,
    const std::string& question) {

	chat_context.push_back(trim_trailing(question));
	std::string prompt = trim_trailing(format_prompt_string());
	prompt += "\n<|im_start|>assistant\n";
	//std::cout << "prompt: " << prompt << std::endl;

	int n_keep = 0;  // Will be set after tokenizing system prompt

	prepare_inputs(client, model_name, RequestType::COMPLETIONS, "chat-123", prompt,
		0.7f, 40, 0.95f, n_keep, -1, true, "<|im_end|>", true, std::vector<float>(), 1.1f);

	process_result(RequestType::COMPLETIONS);
}

// Task abort function (request type 6)
void task_abort(const std::unique_ptr<kc::KyTensorClient>& client,
                const std::string& model_name) {
	prepare_inputs(client, model_name, RequestType::TASK_ABORT, "chat-123", "",  0.0f, 0, 0.0f, 0, 0,
		false, "\n### Human:", true, std::vector<float>(),  // Empty lora scale
		1.0f);

	process_result(RequestType::TASK_ABORT);
}

int main(int argc, char** argv) {
	bool verbose = false;
	std::string url("localhost");

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

	// Register signal handler for Ctrl+C
	signal(SIGINT, signal_handler);

	std::cout << "Connecting to: " << url << std::endl;

	// Create a InferenceServerGrpcClient instance to communicate with the server using gRPC protocol.
	std::unique_ptr<kc::KyTensorClient> client;
	FAIL_IF_ERR(
		kc::KyTensorClient::Create(&client, verbose),
		"unable to create kytensor client");

	std::string model_name = "llamacpp";

	// Start the stream with callback
	FAIL_IF_ERR(
		client->StartStream(
			callback,
			false /*ship_stats*/),  // Don't ship stats
		"unable to establish a streaming connection to server");

	// Get token count for the system prompt and set N_Keep
	tokenize(client, model_name, system_prompt);

	// Main chat loop
	std::cout << "Starting chat interface. Type 'exit' or 'quit' to quit." << std::endl;
	std::string question;
	while (!should_exit) {
		std::cout << "> ";
		if (!std::getline(std::cin, question)) {
			break;
		}

		// Check for exit commands
		if (question == "exit" || question == "quit") {
			break;
		}

		if (question.empty()) {
			continue;
		}

		// Process the chat completion
		chat_completion(client, model_name, question);

		// Keep only the last 2 exchanges (4 entries) to avoid context getting too long
		if (chat_context.size() >= 4) {
			chat_context.erase(chat_context.begin(), chat_context.begin() + 2);
		}
	}

	// Stop the stream before exiting
	FAIL_IF_ERR(client->StopStream(), "unable to stop stream");

	std::cout << "Client exiting..." << std::endl;

	return 0;
}