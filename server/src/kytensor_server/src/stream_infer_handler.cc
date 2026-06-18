#include "stream_infer_handler.h"
#include "infer_response.h"
#include "log.h"

namespace kytensor { namespace server {

using State =
    InferHandlerState<inference::ModelInferRequest, ModelInferRequestPtr, inference::ModelStreamInferResponse>;

// Make sure to keep InferResponseAlloc and OutputBufferQuery logic in sync
TRITONSERVER_Error*
StreamInferResponseAlloc(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  AllocPayload<inference::ModelStreamInferResponse>* payload =
      reinterpret_cast<AllocPayload<inference::ModelStreamInferResponse>*>(
          userp);

  auto response = payload->response_queue_->GetLastAllocatedResponse();

  if (response == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "Unable to access the last allocated response");
  }

  return ResponseAllocatorHelper(
      allocator, tensor_name, byte_size, preferred_memory_type,
      preferred_memory_type_id, response->mutable_infer_response(),
      payload->shm_map_, buffer, buffer_userp, actual_memory_type,
      actual_memory_type_id);
}

TRITONSERVER_Error*
StreamInferResponseStart(TRITONSERVER_ResponseAllocator* allocator, void* userp)
{
  AllocPayload<inference::ModelStreamInferResponse>* payload =
      reinterpret_cast<AllocPayload<inference::ModelStreamInferResponse>*>(
          userp);

  // Move to the next response object
  payload->response_queue_->AllocateResponse();

  return nullptr;  // success
}

TRITONSERVER_Error*
StreamOutputBufferQuery(
    TRITONSERVER_ResponseAllocator* allocator, void* userp,
    const char* tensor_name, size_t* byte_size,
    TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id)
{
  AllocPayload<inference::ModelStreamInferResponse>* payload =
      reinterpret_cast<AllocPayload<inference::ModelStreamInferResponse>*>(
          userp);
  return OutputBufferQueryHelper(
      allocator, tensor_name, byte_size, payload->shm_map_, memory_type,
      memory_type_id);
}

TRITONSERVER_Error*
StreamOutputBufferAttributes(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    TRITONSERVER_BufferAttributes* buffer_attributes, void* userp,
    void* buffer_userp)
{
  AllocPayload<inference::ModelStreamInferResponse>* payload =
      reinterpret_cast<AllocPayload<inference::ModelStreamInferResponse>*>(
          userp);

  return OutputBufferAttributesHelper(
      allocator, tensor_name, payload->shm_map_, buffer_attributes);
}

// stream infer handler implementation

StreamInferHandler::StreamInferHandler(
    const std::string& name,
    const std::shared_ptr<TRITONSERVER_Server>& tritonserver,
    const std::shared_ptr<triton::server::SharedMemoryManager>& shm_manager,
    KytensorCodec* codec,
    std::unordered_map<struct bufferevent*, std::shared_ptr<Connection>>* conns)
{
    FAIL_IF_ERR(
        TRITONSERVER_ResponseAllocatorNew(
            &allocator_, StreamInferResponseAlloc, InferResponseFree,
            StreamInferResponseStart),
        "creating response allocator");
    FAIL_IF_ERR(
        TRITONSERVER_ResponseAllocatorSetQueryFunction(
            allocator_, StreamOutputBufferQuery),
        "setting allocator's query function");
    FAIL_IF_ERR(
        TRITONSERVER_ResponseAllocatorSetBufferAttributesFunction(
            allocator_, StreamOutputBufferAttributes),
        "setting allocator's output buffer attribute query function");

    tritonserver_ = tritonserver;
    codec_ = codec;
    conns_ptr_ = conns;
}

StreamInferHandler::~StreamInferHandler()
{
    for (State* state : state_bucket_) {
        delete state;
    }
    state_bucket_.clear();

    if (allocator_ != nullptr) {
        TRITONSERVER_ResponseAllocatorDelete(allocator_);
        allocator_ = nullptr;
    }
}

State* StreamInferHandler::StateNew(
    TRITONSERVER_Server* tritonserver,
    const std::shared_ptr<StateContext>& context,
    Steps start_step)
{
    State* state = nullptr;

    if (max_state_bucket_count_ > 0) {
        std::lock_guard<std::mutex> lock(alloc_mu_);

        if (!state_bucket_.empty()) {
            state = state_bucket_.back();
            state->Reset(context, start_step);
            state_bucket_.pop_back();
        }
    }

    if (state == nullptr) {
        state = new State(tritonserver, context, start_step);
    }

    context->InsertState(state);

    LOGF_DBG("StateNew, id: %ld, step: %d\n", state->unique_id_, state->step_);
    return state;
}

void StreamInferHandler::StateRelease(State* state)
{
    LOGF_DBG("StateRelease, id: %ld, step: %d\n", state->unique_id_, state->step_);
    if (max_state_bucket_count_ > 0) {
        std::lock_guard<std::mutex> lock(alloc_mu_);

        if (state_bucket_.size() < max_state_bucket_count_) {
            state->Release();
            state_bucket_.push_back(state);
            return;
        }
    }

    delete state;
}

void
StreamInferHandler::OnModelStreamInfer(
    const bufferevent* bev, const MessageHeader& header,
    const ModelInferRequestPtr& request, int64_t receive_time)
{
    TRITONSERVER_Error* err = nullptr;
    int64_t requested_model_version;

    auto context = std::make_shared<StateContext>();
    State* state = StateNew(tritonserver_.get(), context);

    std::lock_guard<std::recursive_mutex> lock(state->step_mtx_);
    auto response_queue = state->response_queue_;

    if (conns_ptr_ != nullptr) {
        auto it = conns_ptr_->find(const_cast<bufferevent *>(bev));
        if (it != conns_ptr_->end()) {
            state->conn_ = it->second;
        }
    }

    bev_ = state->bev_ = const_cast<bufferevent*>(bev);;
    msg_id_ = state->msg_id_ = header.msg_id;
    state->msg_infer_type_ = header.msg_infer_type;
    state->request_ = std::move(request);
    state->handler_ = this;
    state->codec_ = codec_;

    if (err == nullptr) {
        err = GetModelVersionFromString(request->model_version(), &requested_model_version);
    }

    if (err == nullptr) {
        uint32_t txn_flags;
        err = TRITONSERVER_ServerModelTransactionProperties(
            tritonserver_.get(), request->model_name().c_str(),
            requested_model_version, &txn_flags, nullptr);
        if (err == nullptr) {
            state->is_decoupled_ = ((txn_flags & TRITONSERVER_TXN_DECOUPLED) != 0);
        }
    }

    state->context_->IncrementRequestCounter();
    if (!state->is_decoupled_) {
        state->context_->EnqueueForResponse(state);
    }

    //create the inference request
    TRITONSERVER_InferenceRequest* irequest = nullptr;
    if (err == nullptr) {
        err = TRITONSERVER_InferenceRequestNew(
            &irequest, tritonserver_.get(), request->model_name().c_str(),
            requested_model_version);
    }

    if (err == nullptr) {
        state->inference_request_ = {
            irequest, [](TRITONSERVER_InferenceRequest* request) {
                TRITONSERVER_InferenceRequestDelete(request);
                LOGF_DBG("Kytensor: deleted inference request\n");
            }};
        err  = SetInferenceRequestMetadata(irequest, *state->request_, state->parameters_);
    }
    std::list<std::string> serialized_data;

     //convert *request to infer_request
     if (err == nullptr) {
        err = InferKytensorToInput(
            tritonserver_, shm_manager_, *state->request_, &serialized_data, irequest);
    }
    if (err == nullptr) {
        err = InferAllocatorPayload<inference::ModelStreamInferResponse>(
            tritonserver_, shm_manager_, *state->request_, std::move(serialized_data),
            response_queue, &state->alloc_payload_);
    }

    //register request release callback and response callback
    auto request_release_payload =
        std::make_unique<RequestReleasePayload>(state->inference_request_);
    bool response_callback_registered = false;
    if (err == nullptr) {
        err = TRITONSERVER_InferenceRequestSetReleaseCallback(
            irequest, InferRequestComplete,
            request_release_payload.get());
        if (err == nullptr) {
            request_release_payload.release();
        }
    }
    if (err == nullptr) {
        err = TRITONSERVER_InferenceRequestSetResponseCallback(
            irequest, allocator_, &state->alloc_payload_, StreamInferResponseComplete, reinterpret_cast<void*>(state));
        if (err == nullptr) {
            response_callback_registered = true;
        }
    }

    // Get request ID for logging in case of error.
    const char* request_id = "";
    if (irequest != nullptr) {
        TRITONSERVER_Error* err = TRITONSERVER_InferenceRequestId(irequest, &request_id);
        if (err != nullptr) {
            LOGF_ERR("Kytensor: can not getting request id from inference request\n");
            request_id = "<id_unknown>";
            TRITONSERVER_ErrorDelete(err);
        }

    }

    if (!strncmp(request_id, "", 1)) {
        request_id = "<id_unknown>";
    }

    if (err == nullptr) {
        LOGF_DBG(
            "Kytensor: model '%s' version '%s' for request id '%s'\n",
            request->model_name().c_str(), request->model_version().c_str(),
            request_id);
        err = TRITONSERVER_ServerInferAsync(
            tritonserver_.get(), irequest, nullptr /* trace */);
    }

    if (err == nullptr) {
        context->InsertInflightState(state);
    } else {
        LOGF_ERR(
            "Kytensor: failed to create inference request for model '%s' version "
            "'%s' for request id '%s': %s\n",
            request->model_name().c_str(), request->model_version().c_str(),
            request_id, TRITONSERVER_ErrorMessage(err));

        if (!response_callback_registered) {
            MessageHeader err_header;
            err_header.msg_id = header.msg_id;
            err_header.msg_type = MESSAGE_TYPE_RESPONSE;
            err_header.msg_infer_type = header.msg_infer_type;
            err_header.status = TRITONSERVER_ErrorCode(err) + 1;
            inference::ModelInferResponse response;
            codec_->ServerSendMessage(state->conn_, err_header, response);

            state->context_->EraseState(state);
            StateRelease(state);
        }
        TRITONSERVER_ErrorDelete(err);
    }

}

void
StreamInferHandler::StreamInferResponseComplete(
    TRITONSERVER_InferenceResponse* iresponse, const uint32_t flags,
    void* userp)
{
    State* state = reinterpret_cast<State*>(userp);
    LOGF_DBG("Kytensor: StreamInferResponseComplete called for response %p, flags: %d, userp: %p\n", iresponse, flags, userp);

    std::lock_guard<std::recursive_mutex> step_lock(state->step_mtx_);

    uint32_t response_index = state->cb_count_++;
    bool is_complete = ((flags & TRITONSERVER_RESPONSE_COMPLETE_FINAL) != 0);

    TRITONSERVER_Error* err = nullptr;
    auto& response_queue = state->response_queue_;
    inference::ModelStreamInferResponse* response = nullptr;
    if (iresponse) {
        response = response_queue->GetResponseAt(response_index);
        if (response) {
            inference::ModelInferResponse& infer_response =
                *(response->mutable_infer_response());
            err = InferResponseCompleteCommon<inference::ModelStreamInferResponse>(
                state->tritonserver_, iresponse, infer_response, state->alloc_payload_);
        } else {
            LOGF_ERR("failed to get the response object for the inference");
        }

        // Delete the inference response to drop the model reference count
        TRITONSERVER_Error* delete_err =
            TRITONSERVER_InferenceResponseDelete(iresponse);
        if (delete_err != nullptr) {
            LOGF_ERR("failed to delete inference response: %s\n",
                     TRITONSERVER_ErrorMessage(delete_err));
            TRITONSERVER_ErrorDelete(delete_err);
        }
    }

    const bool empty_final = !iresponse &&  state->is_decoupled_ && is_complete;
    const bool enable_empty_final =
        state->parameters_.enable_empty_final_response_;

    const bool create_empty_response = (empty_final && enable_empty_final);
    if (create_empty_response) {
        LOGF_DBG("create empty final response for the inference");
        err = TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            "failed to get the response object for the inference");
        response = new inference::ModelStreamInferResponse();
    }

    if (err != nullptr) {
        response->Clear();
    }

    if (response) {
        auto& infer_response = *(response->mutable_infer_response());
        if (create_empty_response) {
            infer_response.set_id(state->request_->id());
            infer_response.set_model_name(state->request_->model_name());
            infer_response.set_model_version(state->request_->model_version());
        }
        auto& params = *(infer_response.mutable_parameters());
        params["triton_final_response"].set_bool_param(is_complete);
    } else {
        return;
    }

    // Send the response back to the client
    uint8_t err_code;
    if (err == nullptr) {
        err_code = 0;
    } else {
        err_code = TRITONSERVER_ErrorCode(err) + 1;
        TRITONSERVER_ErrorDelete(err);
    }

    //LOGF_INF("server response: %s\n", response->DebugString().c_str());
    MessageHeader header;
    header.msg_id = state->msg_id_;
    header.msg_type = MESSAGE_TYPE_RESPONSE;
    header.msg_infer_type = state->msg_infer_type_;
    header.status = err_code;
    state->codec_->ServerSendMessage(state->conn_, header, *response);

    if (create_empty_response) {
        delete response;
    }

    if (state->is_decoupled_) {
        response->mutable_infer_response()->Clear();
        state->response_queue_->PopResponse();
    }

    if (is_complete) {
        state->context_->EraseInflightState(state);
        state->context_->EraseState(state);
        auto infer_handler = static_cast<StreamInferHandler*>(state->handler_);
        infer_handler->StateRelease(state);
    }
}

}}