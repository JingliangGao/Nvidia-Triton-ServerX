#pragma once

#include <queue>

#include "triton/backend/backend_common.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/core/tritonbackend.h"
#include "triton/core/tritonserver.h"
#include "model_state.h"

//#ifdef TRITON_ENABLE_METRICS
#include "llamacpp_metrics.h"
#include <chrono>
//#endif

namespace triton::backend::llamacpp
{

enum RequestType {
    REQUEST_TYPE_COMPLETION,
	REQUEST_TYPE_OPENAI_COMPLETION,
	REQUEST_TYPE_OPENAI_CHAT,
    REQUEST_TYPE_TOKENIZE,
    REQUEST_TYPE_DETOKENIZE,
	REQUEST_TYPE_LORA_SET,
	REQUEST_TYPE_LORA_GET,
	REQUEST_TYPE_TASK_CANCEL,	//generate cancle
	REQUEST_TYPE_ABORT,   //all inference task termination
	REQUEST_TYPE_METRICS,
	REQUEST_TYPE_WORKER_SAVE,
	REQUEST_TYPE_WORKER_RESTORE,
	REQUEST_TYPE_WORKER_ERASE,
};

struct RequestData {
	TRITONBACKEND_ResponseFactory* factory;
    TRITONBACKEND_Request* tritonRequest;
    std::string tritonRequestId;
	RequestType type;
	int task_id;
	bool stream = false;
};

struct RequestOthers {
	TRITONBACKEND_Request* tritonRequest;
	int request_type;
	json data;
};

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each
// TRITONBACKEND_ModelInstance. ModelInstanceState is derived from
// BackendModelInstance class provided in the backend utilities that
// provides many common functions.
//
class ModelInstanceState {
public:
	static TRITONSERVER_Error* Create(
		ModelState* model_state,
		TRITONBACKEND_ModelInstance* triton_model_instance,
		ModelInstanceState** state);
	virtual ~ModelInstanceState();

	// Get the state of the model that corresponds to this instance.
	ModelState* StateForModel() const { return model_state_; }

	void request_enqueue(TRITONBACKEND_Request** requests, uint32_t const request_count);
	int engine_init();

private:
	ModelInstanceState(
		ModelState* model_state,
		TRITONBACKEND_ModelInstance* triton_model_instance)
		: 	model_state_(model_state),
			llama_engine_(new LlamaCppEngine),
			params_(new common_params)
	{
#ifdef TRITON_ENABLE_METRICS
		llamacpp_metrics_ = std::make_unique<LlamaCppMetrics>();
		llamacpp_metrics_->InitLlamaCppMetrics(model_state->GetModelName(), model_state->GetModelVersion());
#endif
	}

	/// @brief Send a response during enqueue
    void sendEnqueueResponse(TRITONBACKEND_Request* request, TRITONSERVER_Error* error);

	ModelState* model_state_;

	private:
	LlamaCppEngine *llama_engine_;
	std::unique_ptr<common_params> params_;

	//run llamacpp engine start_loop
	std::thread llama_task_thread_;
	//handle inferrence task response thread
	std::thread llama_response_thread_;
	//handle non-inferrence task response thread
	std::thread llama_others_thread_;
	std::thread llama_metrics_thread_;

	std::unordered_map<int, RequestData> taskid_requestdata_;
	std::mutex taskid_requestdata_mutex_;

	std::queue<RequestOthers> others_queue_;
	std::mutex others_mutex_;
	std::condition_variable condition_others;

	const char * device_name_ = nullptr;
	int8_t device_type_;

	//handle inferrence task response
	void wait_for_response();
	bool stop_wait_for_response_ = false;
	void wait_for_others_response();
	bool stop_wait_for_others_ = false;
	void wait_for_metrics();
	bool stop_wait_for_metrics_ = false;

	int handle_completion_request(TRITONBACKEND_Request* request, OAICompatType type);
	int handle_worker_save(TRITONBACKEND_Request* request);
	int handle_worker_restore(TRITONBACKEND_Request* request);
	int handle_worker_erase(TRITONBACKEND_Request* request);
	void handle_queue_tasks(TRITONBACKEND_Request* request, RequestType type);

	void handle_noqueue_tasks(TRITONBACKEND_Request* request, RequestType type);
	void handle_tokenize(RequestOthers &others);
	void handle_detokenize(RequestOthers &others);
	void handle_metrics(RequestOthers &others);
	void handle_lora_set(RequestOthers &others);
	void handle_lora_get(RequestOthers &others);
	void handle_task_cancel(RequestOthers &others);
	void handle_abort(RequestOthers &others);

//#ifdef TRITON_ENABLE_METRICS
    std::unique_ptr<LlamaCppMetrics> llamacpp_metrics_;
//#endif
};

} //namespace triton::backend::llamacpp
