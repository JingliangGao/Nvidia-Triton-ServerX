#include <fstream>
#include "triton/backend/backend_common.h"
#include "model_instance_state.h"
#include "llamacpp_utils.h"
#include "llamacpp_queue.h"
#include "llamacpp_common.h"
#include "utils.h"

namespace triton::backend::llamacpp
{

static std::string read_file(const std::string & fname)
{
	std::ifstream file(fname);
	if (!file) {
		throw std::runtime_error(string_format("error: failed to open file '%s'\n", fname.c_str()));
	}
	std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	file.close();
	return content;
}

ModelInstanceState::~ModelInstanceState()
{
	llama_engine_->queue_tasks.terminate();
	if (llama_task_thread_.joinable()) {
		llama_task_thread_.join();
	}

	stop_wait_for_response_ = true;
	llama_engine_->queue_results.recv_terminate();
	if (llama_response_thread_.joinable()) {
		llama_response_thread_.join();
	}

	stop_wait_for_others_ = true;
	condition_others.notify_one();
	if (llama_others_thread_.joinable()) {
		llama_others_thread_.join();
	}

	llama_backend_free();
	if (llama_engine_)
		delete llama_engine_;
}

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
    try {
      *state = new ModelInstanceState(model_state, triton_model_instance);
    }
    catch (const BackendModelInstanceException& ex) {
      RETURN_ERROR_IF_TRUE(
          ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
          std::string("unexpected nullptr in BackendModelInstanceException"));
      RETURN_IF_ERROR(ex.err_);
    }

    return nullptr;  // success
}

int ModelInstanceState::engine_init()
{
	params_->cpuparams.n_threads = model_state_->n_threads_;
	params_->n_predict = model_state_->n_predict_;
	params_->n_ctx = model_state_->n_ctx_;
	params_->n_batch = model_state_->n_batch_;
	params_->n_keep = 0;
	params_->n_parallel = model_state_->n_parallel_;
	params_->model = model_state_->model_;
	params_->cont_batching = model_state_->cont_batching_;
	params_->flash_attn = true;

	for (const std::string& path : model_state_->lora_path_) {
		params_->lora_adapters.push_back({
			path,
			0.0,
			nullptr,
		});
	}
	params_->lora_init_without_apply = model_state_->lora_init_without_apply_;
	params_->decrypt_type = static_cast<gguf_decrypt_type>(model_state_->model_decrypt_type_);
	params_->slot_save_path = model_state_->worker_save_path_;

	//chat template
	params_->use_jinja = model_state_->use_jinja_;
	if (model_state_->chat_template_file_ != "")
		params_->chat_template = read_file(model_state_->chat_template_file_);

	bool server_verbose = model_state_->llamacpp_verbose_;
	if(server_verbose) {
		params_->verbosity = INT32_MAX;
		common_log_set_verbosity_thold(params_->verbosity);
	}

	// dynamic loading of ggml backend
	ggml_backend_load_all();

	//if find GPU, priority to GPU
	for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
		auto * dev = ggml_backend_dev_get(i);
		device_name_ = ggml_backend_dev_name(dev);
		device_type_ = (int8_t)ggml_backend_dev_type(dev);
		LOG_INF("device %ld: %s type: %d\n", i, ggml_backend_dev_name(dev), ggml_backend_dev_type(dev));
		if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
			params_->n_gpu_layers = 999;
			params_->split_mode = LLAMA_SPLIT_MODE_NONE;
			break;
		}
	}

	//PONN0 not support cache reuse
	auto * dev = ggml_backend_dev_by_name("PONN0");
	if (dev == nullptr) {
		params_->n_cache_reuse = 64;
	}
	llama_engine_->worker_prompt_similarity = 0.4f;

	common_init();

	const char* home_env = std::getenv("HOME");
	std::string log_path = (home_env != nullptr) ? std::string(home_env) + "/.log/kytensor-llama.log" : "./kytensor-llama.log";
	common_log_set_file(common_log_main(), log_path.c_str());

	llama_backend_init();
	llama_numa_init(params_->numa);

	bool ret = llama_engine_->load_model(*params_);
	if (ret == false) {
		return -1;
	}

	llama_engine_->init();

	LOG_INF("model loaded, engine init.\n");

	// print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(*params_).c_str());
        LOG_INF("\n");
    }

	// print sample chat example to make it clear which template is used
    LOG_INF("%s: chat template, chat_template: %s, example_format: '%s'\n", __func__,
        common_chat_templates_source(llama_engine_->chat_templates.get()),
        common_chat_format_example(llama_engine_->chat_templates.get(), llama_engine_->params_base.use_jinja).c_str());

	llama_engine_->queue_tasks.on_new_task(std::bind(
        &LlamaCppEngine::process_single_task, llama_engine_, std::placeholders::_1));
    llama_engine_->queue_tasks.on_update_workers(std::bind(
        &LlamaCppEngine::update_workers, llama_engine_));

	llama_task_thread_ = std::thread([this]() {
		llama_engine_->queue_tasks.start_loop();
	});

	llama_response_thread_ = std::thread([this]() {
		wait_for_response();
	});

	llama_others_thread_ = std::thread([this]() {
		wait_for_others_response();
	});

	// llama_metrics_thread_ = std::thread([this]() {
	// 	wait_for_metrics();
	// });

	return 0;
}

void ModelInstanceState::sendEnqueueResponse(TRITONBACKEND_Request* request, TRITONSERVER_Error* error)
{
    TRITONBACKEND_ResponseFactory* factory;
    LOG_IF_ERROR(TRITONBACKEND_ResponseFactoryNew(&factory, request), "failed to create triton response factory");
    TRITONBACKEND_Response* tritonResponse;
    LOG_IF_ERROR(TRITONBACKEND_ResponseNewFromFactory(&tritonResponse, factory), "Failed to create response");
    LOG_IF_ERROR(TRITONBACKEND_ResponseSend(tritonResponse, TRITONSERVER_RESPONSE_COMPLETE_FINAL, error),
        "Cannot send response");
    LOG_IF_ERROR(TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL), "Cannot release request");
	//why not need TRITONBACKEND_ResponseFactoryDelete
}

int ModelInstanceState::handle_completion_request(TRITONBACKEND_Request* request, OAICompatType type)
{
	json data;
	std::string prompt;
	int32_t n_keep;
	int32_t n_predict;
	float temperature;
	int32_t top_k;
	float top_p;
	bool cache_prompt;
	bool stream;
	std::string stop;
	std::string request_id;
	std::string system_prompt;
	float repeat_penalty;

	request_id = utils::get_request_prompt(request, "request_id");
	prompt = utils::get_request_prompt(request, "text_input");
	auto &params = llama_engine_->params_base;

	if (type == OAICOMPAT_TYPE_NONE) {
		//native style
		system_prompt = utils::get_request_prompt(request, "system_input");
		data["prompt"] = prompt;
		if (!system_prompt.empty()) {
			data["system_prompt"] = system_prompt;
		}

		data["cache_prompt"] = utils::get_request_input<bool>(request, "cache_prompt_input", cache_prompt)
						 ? cache_prompt : true;
		data["n_keep"] =  utils::get_request_input<int32_t>(request, "n_keep_input", n_keep)
						 ? n_keep : 0;
		data["n_predict"] =  utils::get_request_input<int32_t>(request, "n_predict_input", n_predict)
						 ? n_predict : 256;
		data["temperature"] =  utils::get_request_input<float>(request, "temperature_input", temperature)
						 ? temperature : 0.8f;
		data["top_k"] = utils::get_request_input<int32_t>(request, "top_k_input", top_k)
						 ? top_k : 40;
		data["top_p"] = utils::get_request_input<float>(request, "top_p_input", top_p)
						 ? top_p : 0.95f;
		data["stream"] = utils::get_request_input<bool>(request, "stream_input", stream)
						 ? stream : true;
		stop = utils::get_request_prompt(request, "stop_input");
		json stop_array = {stop};
		data["stop"] = stop_array;
		data["repeat_penalty"] = utils::get_request_input<float>(request, "repeat_penalty_input", repeat_penalty)
						 ? repeat_penalty : 1.00f;

		//get input lora
		std::vector<float> lora_scale = utils::get_request_lora_scale(request, "lora_scale_input");
		json lora = json::array();
		for (size_t i = 0; i < lora_scale.size(); i++) {
			lora.push_back({{"id", i}, {"scale", lora_scale[i]}});
		}
		data["lora"] = lora;
	} else {
		//openai style
		json openai_data = json::parse(prompt);

		if (type == OAICOMPAT_TYPE_COMPLETION) {
			data = oaicompat_completion_params_parse(openai_data);
		} else {
			data = oaicompat_completion_params_parse(openai_data, params.use_jinja, params.reasoning_format, llama_engine_->chat_templates.get());
		}
	}

	std::vector<ServerTask> tasks;

	std::vector<llama_tokens> tokenized_prompts = tokenize_input_prompts(llama_engine_->vocab, data["prompt"], true, true);
	tasks.reserve(tokenized_prompts.size());
	ServerTask task = ServerTask(ServerTaskType::SERVER_TASK_TYPE_COMPLETION);

	task.id = llama_engine_->queue_tasks.get_new_id();
	task.index = 0;
	task.prompt_tokens = std::move(tokenized_prompts[0]);
	task.params = ServerTask::params_from_json_cmpl(
		llama_engine_->ctx, llama_engine_->params_base, data);
	//openai type
	task.params.oaicompat = type;
	task.id_selected_worker = json_value(data, "id_selected_worker", -1);

	tasks.push_back(task);


	llama_engine_->queue_results.add_waiting_tasks(tasks);
	llama_engine_->queue_tasks.post(tasks);

	return task.id;

}

int ModelInstanceState::handle_worker_save(TRITONBACKEND_Request* request)
{
	std::string filename = utils::get_request_prompt(request, "text_input");
	if (!fs_validate_filename(filename)) {
		LOG_ERR("filename is not valid.\n");
		sendEnqueueResponse(request, TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, "filename is not valid"));
		return -1;
	}

	std::string filepath = params_->slot_save_path + filename;
	ServerTask task(SERVER_TASK_TYPE_WORKER_SAVE);
	task.id = llama_engine_->queue_tasks.get_new_id();
	const std::string request_id = utils::get_request_prompt(request, "request_id");
	task.worker_action.worker_id = request_id.empty() ? -1 : std::stoi(request_id);
	task.worker_action.filename = filename;
	task.worker_action.filepath = filepath;

	llama_engine_->queue_results.add_waiting_task_id(task.id);
	llama_engine_->queue_tasks.post(task);

	return task.id;

}

int ModelInstanceState::handle_worker_restore(TRITONBACKEND_Request* request)
{
	std::string filename = utils::get_request_prompt(request, "text_input");
	if (!fs_validate_filename(filename)) {
		LOG_ERR("filename is not valid.\n");
		sendEnqueueResponse(request, TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, "filename is not valid"));
		return -1;
	}

	std::string filepath = params_->slot_save_path + filename;
	ServerTask task(SERVER_TASK_TYPE_WORKER_RESTORE);
	const std::string request_id = utils::get_request_prompt(request, "request_id");
	task.id = llama_engine_->queue_tasks.get_new_id();
	task.worker_action.worker_id = request_id.empty() ? -1 : std::stoi(request_id);
	task.worker_action.filename = filename;
	task.worker_action.filepath = filepath;

	llama_engine_->queue_results.add_waiting_task_id(task.id);
	llama_engine_->queue_tasks.post(task);

	return task.id;
}

int ModelInstanceState::handle_worker_erase(TRITONBACKEND_Request* request)
{
	std::string worker_id = utils::get_request_prompt(request, "request_id");
	ServerTask task(SERVER_TASK_TYPE_WORKER_ERASE);
	task.id = llama_engine_->queue_tasks.get_new_id();
	task.worker_action.worker_id = worker_id.empty() ? -1 : std::stoi(worker_id);

	llama_engine_->queue_results.add_waiting_task_id(task.id);
	llama_engine_->queue_tasks.post(task);

	return task.id;
}

void ModelInstanceState::handle_queue_tasks(TRITONBACKEND_Request* request, RequestType type)
{
	int id_task = -1;
	switch (type) {
		case REQUEST_TYPE_COMPLETION:
			id_task = handle_completion_request(request, OAICOMPAT_TYPE_NONE);
			break;
		case REQUEST_TYPE_OPENAI_COMPLETION:
			id_task = handle_completion_request(request, OAICOMPAT_TYPE_COMPLETION);
			break;
		case REQUEST_TYPE_OPENAI_CHAT:
			id_task = handle_completion_request(request, OAICOMPAT_TYPE_CHAT);
			break;
		case REQUEST_TYPE_WORKER_SAVE:
			id_task = handle_worker_save(request);
			break;
		case REQUEST_TYPE_WORKER_RESTORE:
			id_task = handle_worker_restore(request);
			break;
		case REQUEST_TYPE_WORKER_ERASE:
			id_task = handle_worker_erase(request);
			break;
		default:
			break;
	}

	if (id_task < 0) {
		LOG_ERR("task id is not valid.\n");
		sendEnqueueResponse(request, TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, "task id is not valid"));
		return;
	}

	const std::string request_id = utils::get_request_prompt(request, "request_id");
	char const* charRequestId = nullptr;
	TRITONBACKEND_RequestId(request, &charRequestId);
	std::string tritonRequestId;
	if (charRequestId != nullptr)
	{
		tritonRequestId = request_id;
	}
	std::lock_guard<std::mutex> lock(taskid_requestdata_mutex_);
	TRITONBACKEND_ResponseFactory* factory;
	LOG_IF_ERROR(
		TRITONBACKEND_ResponseFactoryNew(&factory, request), "failed to create triton response factory");
	taskid_requestdata_.emplace(id_task,
		RequestData{factory, request, tritonRequestId, type, id_task, true});
}

void ModelInstanceState::handle_noqueue_tasks(TRITONBACKEND_Request* request, RequestType type)
{
	RequestOthers others;
	std::string content;
	others.request_type = type;
	others.tritonRequest = request;
	switch (type) {
		case REQUEST_TYPE_TOKENIZE:
		case REQUEST_TYPE_DETOKENIZE:
			others.data["content"] = utils::get_request_prompt(request, "text_input");
			break;
		case REQUEST_TYPE_LORA_GET:
			break;
		case REQUEST_TYPE_TASK_CANCEL:
			others.data["request_id"] = utils::get_request_prompt(request, "request_id");
			break;
		case REQUEST_TYPE_ABORT:
			others.data["request_id"] = utils::get_request_prompt(request, "request_id");
			break;
		case REQUEST_TYPE_METRICS:
			break;
		default:
			break;
	}

	std::unique_lock<std::mutex> lock(others_mutex_);
	others_queue_.push(others);
	condition_others.notify_one();
}

void ModelInstanceState::request_enqueue(TRITONBACKEND_Request** requests, uint32_t const request_count)
{
	int32_t request_type;
	for (uint32_t i = 0; i < request_count; i++) {
		TRITONBACKEND_Request* request = requests[i];

		utils::get_request_input<int32_t>(request, "request_type", request_type);
		if (request_type < 0) {
			sendEnqueueResponse(request, TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL,
				"input request_type must set"));
		}

		try {
			switch (request_type) {
				case REQUEST_TYPE_COMPLETION:
				case REQUEST_TYPE_OPENAI_COMPLETION:
				case REQUEST_TYPE_OPENAI_CHAT:
				case REQUEST_TYPE_WORKER_SAVE:
				case REQUEST_TYPE_WORKER_RESTORE:
				case REQUEST_TYPE_WORKER_ERASE:
					handle_queue_tasks(request, (RequestType)request_type);
					break;
				case REQUEST_TYPE_TOKENIZE:
				case REQUEST_TYPE_DETOKENIZE:
				case REQUEST_TYPE_LORA_GET:
				case REQUEST_TYPE_TASK_CANCEL:
				case REQUEST_TYPE_ABORT:
				case REQUEST_TYPE_METRICS:
					handle_noqueue_tasks(request, (RequestType)request_type);
					break;
				default:
					break;
			}

		}
		catch (std::exception const &e) {
			sendEnqueueResponse(request, TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, e.what()));
		}
	}
}

void ModelInstanceState::wait_for_response()
{
	while(true) {

		//if not result, it will block
		ServerTaskResultPtr result = llama_engine_->queue_results.recv();
		if (stop_wait_for_response_)
			break;

		RequestData request_data;
		{
			std::lock_guard<std::mutex> lock(taskid_requestdata_mutex_);
			if (!taskid_requestdata_.count(result->id)) {
				LOG_ERR("task id is not in queue results.\n");
			}
			request_data = taskid_requestdata_[result->id];
		}

		auto factory = request_data.factory;
		TRITONBACKEND_Response* tritonResponse = nullptr;
    	LOG_IF_ERROR(TRITONBACKEND_ResponseNewFromFactory(&tritonResponse, factory), "Failed to create response");

		if (result->is_error() == false) {
			json data = result->to_json();
			std::string prompt;
			bool stop_flag = true;

			if (request_data.type == REQUEST_TYPE_COMPLETION ||
				request_data.type == REQUEST_TYPE_OPENAI_COMPLETION ||
				request_data.type == REQUEST_TYPE_OPENAI_CHAT) {

				if (data.contains("stop")) {
					//native style
					prompt = data["content"];
					stop_flag = data["stop"];
				} else {
					//openai style
					prompt = data.dump();
					if (data.is_array()) {
						data[0]["choices"][0]["finish_reason"].is_null() ? stop_flag = false : stop_flag = true;
					} else {
						data["choices"][0]["finish_reason"].is_null() ? stop_flag = false : stop_flag = true;
					}
				}
			} else if (request_data.type == REQUEST_TYPE_WORKER_SAVE ||
					   request_data.type == REQUEST_TYPE_WORKER_RESTORE ||
					   request_data.type == REQUEST_TYPE_WORKER_ERASE) {
				prompt = data.dump();
				stop_flag = true; // set stop to true for worker save
			}
			uint32_t len = prompt.size();
			uint32_t total_len = len + sizeof(len);
			std::vector<int64_t> shape{1, total_len};
			TRITONSERVER_DataType data_type = TRITONSERVER_TYPE_BYTES;
			auto prompt_buffer = utils::getResponseBuffer<uint8_t>(tritonResponse, shape, data_type, "text_output");
			//first 4 byte is length of string
			memcpy(prompt_buffer, &len, sizeof(len));
			memcpy((uint8_t *)prompt_buffer+sizeof(len), prompt.c_str(), len);

			{
				std::lock_guard<std::mutex> lock(taskid_requestdata_mutex_);
				if (stop_flag) {
					LOG_IF_ERROR(TRITONBACKEND_ResponseSend(tritonResponse, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
						"Cannot send response");
					LOG_IF_ERROR(TRITONBACKEND_RequestRelease(request_data.tritonRequest, TRITONSERVER_REQUEST_RELEASE_ALL),
						"Cannot release request");
					LOG_IF_ERROR(TRITONBACKEND_ResponseFactoryDelete(factory), "Cannot delete response factory");
					llama_engine_->queue_results.remove_waiting_task_id(result->id);
					taskid_requestdata_.erase(result->id);
				} else {
					LOG_IF_ERROR(TRITONBACKEND_ResponseSend(tritonResponse, 0, nullptr),
					"Cannot send response");
				}
			}

		} else {
			TRITONSERVER_Error* error = nullptr;
			auto* err = dynamic_cast<ServerTaskResultError*>(result.get());
			if (err->err_type == ERROR_TYPE_DECODE_ABORT)
				error = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, err->err_msg.c_str());
			else
				error = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, "llamacpp result error");

			std::lock_guard<std::mutex> lock(taskid_requestdata_mutex_);
			LOG_IF_ERROR(TRITONBACKEND_ResponseSend(tritonResponse, TRITONSERVER_RESPONSE_COMPLETE_FINAL, error),
				"Cannot send response");
			LOG_IF_ERROR(TRITONBACKEND_RequestRelease(request_data.tritonRequest, TRITONSERVER_REQUEST_RELEASE_ALL),
				"Cannot release request");
			LOG_IF_ERROR(TRITONBACKEND_ResponseFactoryDelete(factory), "Cannot delete response factory");
			llama_engine_->queue_results.remove_waiting_task_id(result->id);
			taskid_requestdata_.erase(result->id);
		}
	}
}

void ModelInstanceState::handle_tokenize(RequestOthers &others)
{
	std::vector<llama_token> tokens;
	if (others.data.count("content") != 0) {
		const bool add_special = json_value(others.data, "add_special", false);
		tokens = tokenize_mixed(llama_engine_->vocab, others.data.at("content"), add_special, true);
	}

	TRITONBACKEND_ResponseFactory* factory;
	LOG_IF_ERROR(TRITONBACKEND_ResponseFactoryNew(&factory, others.tritonRequest), "failed to create triton response factory");
	TRITONBACKEND_Response* tritonResponse;
	LOG_IF_ERROR(TRITONBACKEND_ResponseNewFromFactory(&tritonResponse, factory), "Failed to create response");
	std::vector<int64_t> shape{1, (int32_t)tokens.size()};
	TRITONSERVER_DataType data_type = TRITONSERVER_TYPE_INT32;
	auto buffer = utils::getResponseBuffer<int32_t>(tritonResponse, shape, data_type, "token_output");
	std::copy(tokens.begin(), tokens.end(), (int32_t *)buffer);
	LOG_IF_ERROR(TRITONBACKEND_ResponseSend(tritonResponse, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
				"Cannot send response");
	LOG_IF_ERROR(TRITONBACKEND_RequestRelease(others.tritonRequest, TRITONSERVER_REQUEST_RELEASE_ALL),
				"Cannot release request");
	LOG_IF_ERROR(TRITONBACKEND_ResponseFactoryDelete(factory), "Cannot delete response factory");
}

void ModelInstanceState::handle_detokenize(RequestOthers &others)
{
	std::string content;
	if (others.data.count("content") != 0) {
		std::string tokens = others.data.at("content");
		std::istringstream iss(tokens);
		std::vector<llama_token> token_list;
		llama_token token;
		while (iss >> token) {
			token_list.push_back(token);
		}
		content = tokens_to_str(llama_engine_->ctx, token_list.cbegin(), token_list.cend());
	}

	TRITONBACKEND_ResponseFactory* factory;
	LOG_IF_ERROR(TRITONBACKEND_ResponseFactoryNew(&factory, others.tritonRequest), "failed to create triton response factory");
	TRITONBACKEND_Response* tritonResponse;
	LOG_IF_ERROR(TRITONBACKEND_ResponseNewFromFactory(&tritonResponse, factory), "Failed to create response");
	uint32_t len = content.size();
	uint32_t total_len = len + sizeof(len);
	std::vector<int64_t> shape{1, total_len};
	TRITONSERVER_DataType data_type = TRITONSERVER_TYPE_BYTES;
	auto buffer = utils::getResponseBuffer<uint8_t>(tritonResponse, shape, data_type, "text_output");
	//first 4 byte is length of string
	memcpy(buffer, &len, sizeof(len));
	memcpy((uint8_t *)buffer+sizeof(len), content.c_str(), len);
	LOG_IF_ERROR(TRITONBACKEND_ResponseSend(tritonResponse, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
				"Cannot send response");
	LOG_IF_ERROR(TRITONBACKEND_RequestRelease(others.tritonRequest, TRITONSERVER_REQUEST_RELEASE_ALL),
				"Cannot release request");
	LOG_IF_ERROR(TRITONBACKEND_ResponseFactoryDelete(factory), "Cannot delete response factory");
}

void ModelInstanceState::handle_metrics(RequestOthers &others)
{
	const uint64_t n_prompt_tokens_processed = llama_engine_->metrics.n_prompt_tokens_processed;
	const uint64_t t_prompt_processing       = llama_engine_->metrics.t_prompt_processing;
	const double n_prompt_second =  t_prompt_processing ? 1e3 / t_prompt_processing * n_prompt_tokens_processed : 0;

	const uint64_t n_tokens_predicted  = llama_engine_->metrics.n_tokens_predicted;
	const uint64_t t_tokens_generation = llama_engine_->metrics.t_tokens_generation;
	const double n_tokens_predicted_second = t_tokens_generation ? 1e3 / t_tokens_generation * n_tokens_predicted : 0;

	const int32_t kv_cache_used_cells = llama_kv_self_used_cells(llama_engine_->ctx);
	const int32_t kv_cache_tokens_count = llama_kv_self_n_tokens(llama_engine_->ctx);
	const double kv_cache_used_ratio = llama_engine_->n_ctx ? 1. * kv_cache_used_cells / llama_engine_->n_ctx : 0;

	std::string stat_json = "{";
	stat_json.append("\"n_workers\":" + std::to_string(params_->n_parallel)+",");
	stat_json.append("\"device_name\":" + std::string(device_name_)+",");
	stat_json.append("\"device_type\":" + std::to_string(device_type_)+",");
	//stat_json.append("\"Prompt Tokens Total\":" + std::to_string(n_prompt_tokens_processed)+",");
	//stat_json.append("\"Prompt Time\":" + std::to_string(t_prompt_processing)+",");
	stat_json.append("\"Prompt Tokens Per Seconds\":" + std::to_string(n_prompt_second)+",");
	//stat_json.append("\"Generation Tokens Total\":" + std::to_string(n_tokens_predicted)+",");
	//stat_json.append("\"Generation Time\":" + std::to_string(t_tokens_generation)+",");
	stat_json.append("\"Predicted Tokens Seconds\":" + std::to_string(n_tokens_predicted_second)+",");
	stat_json.append("\"KV Cache Usage Ratio\":" + std::to_string(kv_cache_used_ratio)+",");
	stat_json.append("\"KV Cache Tokens\":" + std::to_string(kv_cache_tokens_count)+",");
	stat_json.back() = '}';

//#ifdef TRITON_ENABLE_METRICS
//	LOG_IF_ERROR(llamacpp_metrics_->UpdateLlamaCppMetrics(stat_json), "Failed updating llamacpp statistics");
//#endif

	TRITONBACKEND_ResponseFactory* factory;
	LOG_IF_ERROR(TRITONBACKEND_ResponseFactoryNew(&factory, others.tritonRequest), "failed to create triton response factory");
	TRITONBACKEND_Response* tritonResponse;
	LOG_IF_ERROR(TRITONBACKEND_ResponseNewFromFactory(&tritonResponse, factory), "Failed to create response");
	uint32_t len = strlen(stat_json.c_str());
	uint32_t total_len = len + sizeof(len);
	std::vector<int64_t> shape{1, total_len};
	TRITONSERVER_DataType data_type = TRITONSERVER_TYPE_BYTES;
	auto buffer = utils::getResponseBuffer<uint8_t>(tritonResponse, shape, data_type, "metrics_output");
	//first 4 byte is length of string
	memcpy(buffer, &len, sizeof(len));
	memcpy((uint8_t *)buffer+sizeof(len), stat_json.c_str(), len);
	LOG_IF_ERROR(TRITONBACKEND_ResponseSend(tritonResponse, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr), 
				"Cannot send response");
	LOG_IF_ERROR(TRITONBACKEND_RequestRelease(others.tritonRequest, TRITONSERVER_REQUEST_RELEASE_ALL), 
				"Cannot release request");
	LOG_IF_ERROR(TRITONBACKEND_ResponseFactoryDelete(factory), "Cannot delete response factory");

	//reset metrics
	//llama_engine_->metrics.reset_bucket();
}

#if 0
void ModelInstanceState::handle_lora_set(RequestOthers &others)
{
	std::vector<float> lora_scale = utils::get_request_lora_scale(others.tritonRequest, "lora_scale_input");
	int lora_adapter_size = llama_engine_->lora_adapters.size();
	int size = lora_scale.size();

	if (size != lora_adapter_size) {
		throw std::runtime_error("set lora_adpater size is not equal lora_adapter size");
	}

	for (int i = 0; i < size; i++) {
		llama_engine_->lora_adapters[i].scale = lora_scale[i];
	}

	ServerTask task;
	task.type = SERVER_TASK_TYPE_SET_LORA;
	const int id_task = llama_engine_->queue_tasks.post(task);
	llama_engine_->queue_results.add_waiting_task_id(id_task);

	ServerTaskResult result = llama_engine_->queue_results.recv(id_task);
	llama_engine_->queue_results.remove_waiting_task_id(id_task);

	std::string ret;
	ret = "success";
	TRITONBACKEND_ResponseFactory* factory;
	LOG_IF_ERROR(TRITONBACKEND_ResponseFactoryNew(&factory, others.tritonRequest), "failed to create triton response factory");
	TRITONBACKEND_Response* tritonResponse;
	LOG_IF_ERROR(TRITONBACKEND_ResponseNewFromFactory(&tritonResponse, factory), "Failed to create response");
	uint32_t len = ret.size();
	uint32_t total_len = len + sizeof(len);
	std::vector<int64_t> shape{1, total_len};
	TRITONSERVER_DataType data_type = TRITONSERVER_TYPE_BYTES;
	auto buffer = utils::getResponseBuffer<uint8_t>(tritonResponse, shape, data_type, "lora_scale_output");
	//first 4 byte is length of string
	memcpy(buffer, &len, sizeof(len));
	memcpy((uint8_t *)buffer+sizeof(len), ret.c_str(), len);
	LOG_IF_ERROR(TRITONBACKEND_ResponseSend(tritonResponse, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
				"Cannot send response");
	LOG_IF_ERROR(TRITONBACKEND_RequestRelease(others.tritonRequest, TRITONSERVER_REQUEST_RELEASE_ALL),
				"Cannot release request");
	LOG_IF_ERROR(TRITONBACKEND_ResponseFactoryDelete(factory), "Cannot delete response factory");

}
#endif

void ModelInstanceState::handle_lora_get(RequestOthers &others)
{
	std::string lora_scale;
	const auto & loras = llama_engine_->params_base.lora_adapters;

	for (size_t i = 0; i < loras.size(); ++i) {
		auto & la = loras[i];
		lora_scale += "id:" + std::to_string(i) + ",";
		lora_scale += "path:" + la.path + ",";
		lora_scale += "scale:" + std::to_string(la.scale) + ";";
	}

	//if (llama_engine_->lora_adapters.size() == 0)
	//	lora_scale = "none";

	TRITONBACKEND_ResponseFactory* factory;
	LOG_IF_ERROR(TRITONBACKEND_ResponseFactoryNew(&factory, others.tritonRequest), "failed to create triton response factory");
	TRITONBACKEND_Response* tritonResponse;
	LOG_IF_ERROR(TRITONBACKEND_ResponseNewFromFactory(&tritonResponse, factory), "Failed to create response");
	uint32_t len = lora_scale.size();
	uint32_t total_len = len + sizeof(len);
	std::vector<int64_t> shape{1, total_len};
	TRITONSERVER_DataType data_type = TRITONSERVER_TYPE_BYTES;
	auto buffer = utils::getResponseBuffer<uint8_t>(tritonResponse, shape, data_type, "lora_scale_output");
	//first 4 byte is length of string
	memcpy(buffer, &len, sizeof(len));
	memcpy((uint8_t *)buffer+sizeof(len), lora_scale.c_str(), len);
	LOG_IF_ERROR(TRITONBACKEND_ResponseSend(tritonResponse, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr), 
				"Cannot send response");
	LOG_IF_ERROR(TRITONBACKEND_RequestRelease(others.tritonRequest, TRITONSERVER_REQUEST_RELEASE_ALL), 
				"Cannot release request");
	LOG_IF_ERROR(TRITONBACKEND_ResponseFactoryDelete(factory), "Cannot delete response factory");
}

void ModelInstanceState::handle_task_cancel(RequestOthers &others)
{
	std::string request_id = others.data.at("request_id");
	RequestData request_data;
	int id_task = -1;
	{
		std::lock_guard<std::mutex> lock(taskid_requestdata_mutex_);
		for (auto &it : taskid_requestdata_) {
			if (it.second.tritonRequestId == request_id) {
				id_task = it.first;
				request_data = it.second;
				break;
			}
		}
	}

	if (id_task != -1) {
		auto factory_complete = request_data.factory;

		std::unordered_set<int> id_tasks;
		id_tasks.insert(id_task);
		llama_engine_->cancel_tasks(id_tasks);
		taskid_requestdata_.erase(id_task);

		TRITONBACKEND_Response* tritonResponse_complete;
		LOG_IF_ERROR(TRITONBACKEND_ResponseNewFromFactory(&tritonResponse_complete, factory_complete), "Failed to create response");
		LOG_IF_ERROR(TRITONBACKEND_ResponseSend(tritonResponse_complete, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
				"Cannot send response");
		//release complete request
		LOG_IF_ERROR(TRITONBACKEND_RequestRelease(request_data.tritonRequest, TRITONSERVER_REQUEST_RELEASE_ALL),
						"Cannot release request");
		LOG_IF_ERROR(TRITONBACKEND_ResponseFactoryDelete(factory_complete), "Cannot delete response factory");
	}

	//release cancle request
	TRITONBACKEND_ResponseFactory* factory;
	LOG_IF_ERROR(TRITONBACKEND_ResponseFactoryNew(&factory, others.tritonRequest), "failed to create triton response factory");
	TRITONBACKEND_Response* tritonResponse;
	LOG_IF_ERROR(TRITONBACKEND_ResponseNewFromFactory(&tritonResponse, factory), "Failed to create response");
	LOG_IF_ERROR(TRITONBACKEND_ResponseSend(tritonResponse, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr), 
				"Cannot send response");
	LOG_IF_ERROR(TRITONBACKEND_RequestRelease(others.tritonRequest, TRITONSERVER_REQUEST_RELEASE_ALL), 
				"Cannot release request");
	LOG_IF_ERROR(TRITONBACKEND_ResponseFactoryDelete(factory), "Cannot delete response factory");

}

//handle abort work successful on one worker.
//if worker more than one, it has some problem, for example, in long text prompt,
// one worker in prefill, but another worker don't launch work, it will have problem.
void ModelInstanceState::handle_abort(RequestOthers &others)
{
	std::string request_id = others.data.at("request_id");
	RequestData request_data;
	int id_task = -1;
	{
		std::lock_guard<std::mutex> lock(taskid_requestdata_mutex_);
		for (auto &it : taskid_requestdata_) {
			if (it.second.tritonRequestId == request_id) {
				id_task = it.first;
				request_data = it.second;
				break;
			}
		}
	}

	if (id_task != -1) {

		bool removed = llama_engine_->queue_tasks.abort_clenup_pending_task(id_task);
		if(!removed) {
			llama_engine_->decode_abort_flag_ = true;
		}
	}

	//release cancle request
	TRITONBACKEND_ResponseFactory* factory;
	LOG_IF_ERROR(TRITONBACKEND_ResponseFactoryNew(&factory, others.tritonRequest), "failed to create triton response factory");
	TRITONBACKEND_Response* tritonResponse;
	LOG_IF_ERROR(TRITONBACKEND_ResponseNewFromFactory(&tritonResponse, factory), "Failed to create response");
	LOG_IF_ERROR(TRITONBACKEND_ResponseSend(tritonResponse, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
				"Cannot send response");
	LOG_IF_ERROR(TRITONBACKEND_RequestRelease(others.tritonRequest, TRITONSERVER_REQUEST_RELEASE_ALL),
				"Cannot release request");
	LOG_IF_ERROR(TRITONBACKEND_ResponseFactoryDelete(factory), "Cannot delete response factory");

}

void ModelInstanceState::wait_for_others_response()
{
	while(true) {
		std::unique_lock<std::mutex> lock(others_mutex_);
		condition_others.wait(lock, [&] {
			return !others_queue_.empty() || stop_wait_for_others_;
		});

		if (stop_wait_for_others_)
			break;

		RequestOthers others = others_queue_.front();
		others_queue_.pop();

		if (others.request_type == REQUEST_TYPE_TOKENIZE)
			handle_tokenize(others);
		else if (others.request_type == REQUEST_TYPE_DETOKENIZE)
			handle_detokenize(others);
		//else if (others.request_type == REQUEST_TYPE_LORA_SET)
			//handle_lora_set(others);
		else if (others.request_type == REQUEST_TYPE_LORA_GET)
			handle_lora_get(others);
		else if (others.request_type == REQUEST_TYPE_TASK_CANCEL)
			handle_task_cancel(others);
		else if (others.request_type == REQUEST_TYPE_ABORT)
			handle_abort(others);
		else if (others.request_type == REQUEST_TYPE_METRICS)
			handle_metrics(others);

	}
}

// void ModelInstanceState::wait_for_metrics()
// {
// 	while(!stop_wait_for_metrics_) {
// 		std::this_thread::sleep_for(std::chrono::milliseconds(model_state_->metrics_period_ms_));

// 		// create request task
// 		ServerTask task;
// 		task.id = llama_engine_->queue_tasks.get_new_id();
// 		task.id_multi = -1;
// 		task.id_target = -1;
// 		task.type = SERVER_TASK_TYPE_METRICS;
// 		task.data.push_back({{"reset_bucket", true}});

// 		llama_engine_->queue_results.add_waiting_task_id(task.id);
// 		llama_engine_->queue_tasks.post(task);

// 		// get the result
// 		ServerTaskResult result = llama_engine_->queue_results.recv(task.id);
// 		llama_engine_->queue_results.remove_waiting_task_id(task.id);

// 		json data = result.data;

// 		const uint64_t n_prompt_tokens_processed = data.at("n_prompt_tokens_processed");
//         const uint64_t t_prompt_processing       = data.at("t_prompt_processing");

//         const uint64_t n_tokens_predicted  = data.at("n_tokens_predicted");
//         const uint64_t t_tokens_generation = data.at("t_tokens_generation");

//         const int32_t kv_cache_used_cells = data.at("kv_cache_used_cells");

// 		std::string stat_json = "{";
// 		stat_json.append("\"Prompt Tokens Total\":" + std::to_string((uint64_t)data.at("n_prompt_tokens_processed_total"))+",");
// 		stat_json.append("\"Prompt Seconds Total\":" + std::to_string((uint64_t)data.at("t_prompt_processing_total") / 1.e3)+",");
// 		stat_json.append("\"Tokens Predicted Total\":" + std::to_string((uint64_t)data.at("n_tokens_predicted_total"))+",");
// 		stat_json.append("\"Tokens Predicted Seconds Total\":" + std::to_string((uint64_t)data.at("t_tokens_generation_total") / 1.e3)+",");
// 		stat_json.append("\"Prompt Tokens Seconds\":" + std::to_string(n_prompt_tokens_processed 
// 							? 1.e3 / t_prompt_processing * n_prompt_tokens_processed : 0.)+",");
// 		stat_json.append("\"Predict Tokens Seconds\":" + std::to_string(n_tokens_predicted 
// 							? 1.e3 / t_tokens_generation * n_tokens_predicted : 0.)+",");
// 		stat_json.append("\"KV Cache Usage Ratio\":" + std::to_string(1. * kv_cache_used_cells / llama_engine_->params.n_ctx)+",");
// 		stat_json.append("\"KV Cache Tokens\":" + std::to_string((uint64_t) data.at("kv_cache_tokens_count"))+",");
// 		stat_json.append("\"Requests Processing\":" + std::to_string((uint64_t) data.at("processing"))+",");
// 		stat_json.append("\"Requests Deferred\":" + std::to_string((uint64_t) data.at("deferred"))+",");
// 		stat_json.back() = '}';

// 		LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, stat_json.c_str());
// //#ifdef TRITON_ENABLE_METRICS
// 		LOG_IF_ERROR(llamacpp_metrics_->UpdateLlamaCppMetrics(stat_json), "Failed updating llamacpp statistics");
// //#endif
// 	}
// }

} //namespace triton::backend::llamacpp
