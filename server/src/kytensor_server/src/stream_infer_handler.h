#pragma once
#include <string>
#include <memory>
#include <queue>
#include <atomic>
#include <mutex>
#include <thread>
#include <unordered_map>

#include <event2/event.h>
#include <event2/bufferevent.h>

#include "triton/core/tritonserver.h"
#include "../../shared_memory_manager.h"
#include "kytensor_common.h"
#include "infer_response.h"
#include "infer_handler.h"
#include "kytensor_codec.h"
#include "kytensor_response.h"

namespace kytensor { namespace server {

class StreamInferHandler;

// Make sure to keep InferResponseAlloc and OutputBufferQuery logic in sync
TRITONSERVER_Error* StreamInferResponseAlloc(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id);

//
// Additional Stream Infer utilities
//
TRITONSERVER_Error* StreamInferResponseStart(
    TRITONSERVER_ResponseAllocator* allocator, void* userp);

// Make sure to keep InferResponseAlloc and OutputBufferQuery logic in sync
TRITONSERVER_Error* StreamOutputBufferQuery(
    TRITONSERVER_ResponseAllocator* allocator, void* userp,
    const char* tensor_name, size_t* byte_size,
    TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id);

// Make sure to keep InferResponseAlloc, OutputBufferQuery, and
// OutputBufferAttributes logic in sync
TRITONSERVER_Error* StreamOutputBufferAttributes(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    TRITONSERVER_BufferAttributes* buffer_attributes, void* userp,
    void* buffer_userp);

class StreamInferHandler {
public:
    using State =
        InferHandlerState<inference::ModelInferRequest, ModelInferRequestPtr, inference::ModelStreamInferResponse>;
    using StateContext = typename State::Context;

    StreamInferHandler(
        const std::string& name,
        const std::shared_ptr<TRITONSERVER_Server>& tritonserver,
        const std::shared_ptr<triton::server::SharedMemoryManager>& shm_manager,
        KytensorCodec* codec,
        std::unordered_map<struct bufferevent*, std::shared_ptr<Connection>>* conns);
    ~StreamInferHandler();

    void OnModelStreamInfer(const struct bufferevent* bev,
        const MessageHeader& header,
        const ModelInferRequestPtr& request,
        int64_t receive_time);

State* StateNew(
    TRITONSERVER_Server* tritonserver,
    const std::shared_ptr<StateContext>& context,
    Steps start_step = Steps::START);

void StateRelease(State* state);

private:
    static void StreamInferResponseComplete(
        TRITONSERVER_InferenceResponse* iresponse, const uint32_t flags,
        void* userp);

    std::string name_;
    std::shared_ptr<TRITONSERVER_Server> tritonserver_;
    const struct bufferevent* bev_;
    KytensorCodec* codec_;
    std::unordered_map<struct bufferevent*, std::shared_ptr<Connection>>* conns_ptr_{nullptr};
    uint32_t msg_id_;
    std::shared_ptr<triton::server::SharedMemoryManager> shm_manager_;

    TRITONSERVER_ResponseAllocator* allocator_;

    std::mutex alloc_mu_;
    const size_t max_state_bucket_count_ = 8;
    std::vector<State*> state_bucket_;
    KytensorResponse response_;
};

}}