#include "infer_handler.h"
#include "infer_response.h"
#include "log.h"

namespace kytensor { namespace server {

using State =
    InferHandlerState<inference::ModelInferRequest, ModelInferRequestPtr, inference::ModelInferResponse>;

#ifndef NDEBUG
uint64_t
NextUniqueId()
{
    static std::atomic<uint64_t> id(0);
    return ++id;
}
#endif  // NDEBUG

// Make sure to keep InferResponseAlloc and OutputBufferQuery logic in sync
TRITONSERVER_Error*
InferResponseAlloc(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  AllocPayload<inference::ModelInferResponse>* payload =
      reinterpret_cast<AllocPayload<inference::ModelInferResponse>*>(userp);

  // ModelInfer RPC expects exactly one response per request. Hence,
  // will be creating and using just one response object.
  inference::ModelInferResponse* response =
      payload->response_queue_->GetNonDecoupledResponse();
  return ResponseAllocatorHelper(
      allocator, tensor_name, byte_size, preferred_memory_type,
      preferred_memory_type_id, response, payload->shm_map_, buffer,
      buffer_userp, actual_memory_type, actual_memory_type_id);
}

// Make sure to keep InferResponseAlloc and OutputBufferQuery logic in sync
TRITONSERVER_Error*
OutputBufferQuery(
    TRITONSERVER_ResponseAllocator* allocator, void* userp,
    const char* tensor_name, size_t* byte_size,
    TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id)
{
  AllocPayload<inference::ModelInferResponse>* payload =
      reinterpret_cast<AllocPayload<inference::ModelInferResponse>*>(userp);

  return OutputBufferQueryHelper(
      allocator, tensor_name, byte_size, payload->shm_map_, memory_type,
      memory_type_id);
}

// Make sure to keep InferResponseAlloc, OutputBufferQuery, and
// OutputBufferAttributes logic in sync
TRITONSERVER_Error*
OutputBufferAttributes(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    TRITONSERVER_BufferAttributes* buffer_attributes, void* userp,
    void* buffer_userp)
{
  AllocPayload<inference::ModelInferResponse>* payload =
      reinterpret_cast<AllocPayload<inference::ModelInferResponse>*>(userp);

    return OutputBufferAttributesHelper(
        allocator, tensor_name, payload->shm_map_, buffer_attributes);
    return nullptr;  // Success
}

TRITONSERVER_Error*
InferResponseStart(TRITONSERVER_ResponseAllocator* allocator, void* userp)
{
    AllocPayload<inference::ModelInferResponse>* payload =
        reinterpret_cast<AllocPayload<inference::ModelInferResponse>*>(userp);

    // ModelInfer RPC expects exactly one response per request. Hence, always call
    // GetNonDecoupledResponse() to create one response object on response start.
    payload->response_queue_->GetNonDecoupledResponse();

    return nullptr;  // success
}

TRITONSERVER_Error*
InferResponseFree(
    TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
    LOGF_DBG("Kytesnsor: free buffer %p of size %zu\n", buffer, byte_size);

    // Don't do anything when releasing a buffer since InferResponseAlloc
    // wrote directly into the response protobuf.
    return nullptr;  // Success
}

InferHandler::InferHandler(
    const std::string& name,
    const std::shared_ptr<TRITONSERVER_Server>& tritonserver,
    const std::shared_ptr<triton::server::SharedMemoryManager>& shm_manager,
    KytensorCodec* codec,
    std::unordered_map<struct bufferevent*, std::shared_ptr<Connection>>* conns)
    : name_(name), shm_manager_(shm_manager)
{
    //register response allocator functions
    FAIL_IF_ERR(
        TRITONSERVER_ResponseAllocatorNew(
            &allocator_, InferResponseAlloc, InferResponseFree,
            InferResponseStart),
        "creating inference response allocator");
    FAIL_IF_ERR(
        TRITONSERVER_ResponseAllocatorSetQueryFunction(
            allocator_, OutputBufferQuery),
        "setting allocator's query function");
    FAIL_IF_ERR(
        TRITONSERVER_ResponseAllocatorSetBufferAttributesFunction(
            allocator_, OutputBufferAttributes),
        "setting allocator's output buffer attributes function");

    //save in ctx_
    tritonserver_ = tritonserver;
    codec_ = codec;
    conns_ptr_ = conns;
}

InferHandler::~InferHandler()
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

void
InferHandler::OnModelInfer(
    const struct bufferevent* bev, const MessageHeader& header,
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
        if ((err == nullptr) && (txn_flags & TRITONSERVER_TXN_DECOUPLED)) {
            err = TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_UNSUPPORTED,
                "decoupled model transaction is not supported for kytensor");
        }
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
        err = InferAllocatorPayload<inference::ModelInferResponse>(
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
            irequest, allocator_, &state->alloc_payload_, InferResponseComplete, reinterpret_cast<void*>(state));
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

State*
InferHandler::StateNew(
    TRITONSERVER_Server* tritonserver,
    const std::shared_ptr<StateContext>& context, Steps start_step)
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

void
InferHandler::StateRelease(State* state)
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
InferHandler::InferResponseComplete(
    TRITONSERVER_InferenceResponse* iresponse, const uint32_t flags, void* userp)
{
    State* state = reinterpret_cast<State*>(userp);
    LOGF_DBG("Kytensor: InferResponseComplete called for response %p\n", iresponse);

    std::lock_guard<std::recursive_mutex> step_lock(state->step_mtx_);

    if (iresponse != nullptr) {
        state->cb_count_++;
    }

    // Allow sending 1 response and final flag separately, only mark
    // non-inflight when seeing final flag
    if (flags & TRITONSERVER_RESPONSE_COMPLETE_FINAL) {
        state->context_->EraseInflightState(state);
    }

    TRITONSERVER_Error* err = nullptr;
    inference::ModelInferResponse* response =
        state->response_queue_->GetResponseAt(0);
    bool response_created = false;
    if (response == nullptr) {
        LOGF_ERR("failed to get the response object for the inference");
        err = TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            "failed to get the response object for the inference");
        response_created = true;
        response = new inference::ModelInferResponse();
    }

    if (state->cb_count_ != 1) {
        err = TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL, std::string(
                                             "expected a single response, got " +
                                             std::to_string(state->cb_count_))
                                             .c_str());
    } else if (iresponse != nullptr) {
        err = InferResponseCompleteCommon<inference::ModelInferResponse>(
            state->tritonserver_, iresponse, *response, state->alloc_payload_);
    }

    if (err != nullptr) {
        response->Clear();
    }

    // Delete the inference response to drop the model reference count
    TRITONSERVER_Error* delete_err =
        TRITONSERVER_InferenceResponseDelete(iresponse);
    if (delete_err != nullptr) {
        LOGF_ERR("failed to delete inference response: %s\n",
                 TRITONSERVER_ErrorMessage(delete_err));
        TRITONSERVER_ErrorDelete(delete_err);
    }

    if ((flags & TRITONSERVER_RESPONSE_COMPLETE_FINAL) == 0) {
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

    if (response_created) {
        delete response;
    }

    state->context_->EraseState(state);
    auto infer_handler = static_cast<InferHandler*>(state->handler_);
    infer_handler->StateRelease(state);
}

}}