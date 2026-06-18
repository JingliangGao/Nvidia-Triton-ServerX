#pragma once
#include <string>
#include <memory>
#include <queue>
#include <atomic>
#include <mutex>
#include <thread>

#include <event2/event.h>
#include <event2/bufferevent.h>

#include "triton/core/tritonserver.h"
#include "../../shared_memory_manager.h"
#include "kytensor_common.h"
#include "infer_response.h"
#include "kytensor_codec.h"
#include "kytensor_response.h"

namespace kytensor { namespace server {

// Unique IDs are only needed when debugging. They only appear in
// verbose logging.
#ifndef NDEBUG
uint64_t NextUniqueId();
#define NEXT_UNIQUE_ID NextUniqueId()
#else
#define NEXT_UNIQUE_ID (0)
#endif  // NDEBUG

class InferHandler;

TRITONSERVER_Error* InferResponseFree(
    TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id);
//
// InferHandlerState
//
template <
    typename RequestType, typename RequestTypePtr, typename ResponseType>
class InferHandlerState {
public:
    using InferHandlerStateType =
      InferHandlerState<RequestType, RequestTypePtr, ResponseType>;

    // State that is shared across all state objects that make up a GRPC
    // transaction (e.g. a stream).
    struct Context {
        explicit Context(const uint64_t unique_id = 0)
            : unique_id_(unique_id), ongoing_requests_(0),
            step_(Steps::START), finish_ok_(true), ongoing_write_(false)
        {
        }

        // Increments the ongoing request counter
        void IncrementRequestCounter() { ongoing_requests_++; }

        // Decrements the ongoing request counter
        void DecrementRequestCounter() { ongoing_requests_--; }

        // Adds the state object created on this context
        void InsertState(InferHandlerStateType* state)
        {
            all_states_.insert(state);
        }

        // Erases the state object created on this context
        void EraseState(InferHandlerStateType* state)
        {
            EraseInflightState(state);
            all_states_.erase(state);
        }

        const std::string DebugString(InferHandlerStateType* state)
        {
            std::string debug_string("");
            debug_string.append(
                "Running state_id " + std::to_string(state->unique_id_) + "\n");
            debug_string.append(
                "\tContext step " + std::to_string(state->context_->step_) + " id " +
                std::to_string(state->context_->unique_id_) + "\n");
            for (auto new_state : all_states_) {
                debug_string.append(
                    "\t\t State id " + std::to_string(new_state->unique_id_) +
                    ": State step " + std::to_string(new_state->step_) + "\n");
            }

            return debug_string;
        }

        // Inserts the state to a set tracking active requests
        // within the server core. Should only be called when
        // the request was successfully enqueued on Triton.
        void InsertInflightState(InferHandlerStateType* state)
        {
            std::lock_guard<std::recursive_mutex> lock(mu_);
            inflight_states_.insert(state);
        }

        // Erases the state to a set tracking active requests
        // within the server core.
        void EraseInflightState(InferHandlerStateType* state)
        {
            std::lock_guard<std::recursive_mutex> lock(mu_);
            inflight_states_.erase(state);
        }

        // Enqueue 'state' so that its response is delivered in the
        // correct order.
        void EnqueueForResponse(InferHandlerStateType* state)
        {
        std::lock_guard<std::recursive_mutex> lock(mu_);
        states_.push(state);
        }

        // If 'state' is at the front of the queue and written, pop it and
        // return true. Other return false.
        bool PopCompletedResponse(InferHandlerStateType* state)
        {
            std::lock_guard<std::recursive_mutex> lock(mu_);
            if (states_.empty()) {
                return false;
            }

            InferHandlerStateType* front = states_.front();
            if ((front == state) && (state->step_ == Steps::WRITTEN)) {
                states_.pop();
                return true;
            }

            return false;
        }

        // Return true if this context has completed all reads and writes.
        bool IsRequestsCompleted()
        {
            std::lock_guard<std::recursive_mutex> lock(mu_);
            return (
                (step_ == Steps::WRITEREADY) && states_.empty() &&
                (ongoing_requests_ == 0));
        }

        // Unique ID for the context. Used only for debugging so will
        // always be 0 in non-debug builds.
        const uint64_t unique_id_;

        // The states associated with this context that are currently
        // active. Used by stream handlers to maintain request / response
        // orders. A state enters this queue when it has successfully read
        // a request and exits the queue when it is written.
        std::recursive_mutex mu_;
        std::queue<InferHandlerStateType*> states_;
        std::atomic<uint32_t> ongoing_requests_;

        // Tracks the inflight requests sent to Triton core via this
        // context. We will use this structure to issue cancellations
        // on these requests.
        std::set<InferHandlerStateType*> inflight_states_;

        // Tracks all the states that have been created on this context.
        std::set<InferHandlerStateType*> all_states_;

        // Ready to write queue for decoupled
        std::queue<InferHandlerStateType*> ready_to_write_states_;

        // The step of the entire context.
        Steps step_;

        // True if this context should finish with OK status, false if
        // should finish with CANCELLED status.
        bool finish_ok_;

        // True if there is an ongoing write to the grpc stream
        std::atomic<bool> ongoing_write_;
    };

    // This constructor is used to build a wrapper state object
    // pointing to the actual state object. The wrapper state
    // object is used to distinguish a tag from AsyncNotifyWhenDone()
    // signal.
    explicit InferHandlerState(Steps start_step, InferHandlerState* state)
        : step_(start_step), state_ptr_(state), async_notify_state_(false)
    {
        state->MarkAsAsyncNotifyState();
    }

    explicit InferHandlerState(
        TRITONSERVER_Server* tritonserver,
        const std::shared_ptr<Context>& context, Steps start_step = Steps::START)
        : tritonserver_(tritonserver), async_notify_state_(false)
    {
        // For debugging and testing
        const char* dstr = getenv("TRITONSERVER_DELAY_GRPC_RESPONSE");
        delay_response_ms_ = 0;
        if (dstr != nullptr) {
        delay_response_ms_ = atoi(dstr);
        }
        const char* cstr = getenv("TRITONSERVER_DELAY_GRPC_COMPLETE");
        delay_complete_ms_ = 0;
        if (cstr != nullptr) {
        delay_complete_ms_ = atoi(cstr);
        }
        const char* pstr = getenv("TRITONSERVER_DELAY_GRPC_PROCESS");
        delay_process_ms_ = 0;
        if (pstr != nullptr) {
        delay_process_ms_ = atoi(pstr);
        }

        response_queue_.reset(new InferResponseQueue<ResponseType>());
        Reset(context, start_step);
    }

    ~InferHandlerState() { ClearTraceTimestamps(); }

    bool IsGrpcContextCancelled() { return context_->IsCancelled(); }

    void Reset(
        const std::shared_ptr<Context>& context, Steps start_step = Steps::START)
    {
        unique_id_ = NEXT_UNIQUE_ID;
        context_ = context;
        step_ = start_step;
        status_ = 0;
        cb_count_ = 0;
        is_decoupled_ = false;
        complete_ = false;
        parameters_ = {};
        request_ = nullptr;
        response_queue_->Reset();
        // Clear trace_timestamps_ here so they do not grow indefinitely since
        // states are re-used for performance.
        ClearTraceTimestamps();
        // The pointer should be nullptr for all state objects instead of
        // wrapper state object in WAITING_NOTIFICATION step.
        state_ptr_ = nullptr;
        async_notify_state_ = false;
    }

    void Release()
    {
        context_ = nullptr;
        inference_request_.reset();
        request_ = nullptr;
        conn_.reset();
        bev_ = nullptr;
        msg_id_ = 0;
        msg_infer_type_ = 0;
        cb_count_ = 0;
        handler_ = nullptr;
        ClearTraceTimestamps();
    }

    void ClearTraceTimestamps()
    {
    #ifdef TRITON_ENABLE_TRACING
        if (trace_ != nullptr) {
        for (const auto& timestamp : trace_timestamps_) {
            trace_->CaptureTimestamp(timestamp.first, timestamp.second);
        }
        trace_.reset();
        }
        trace_timestamps_.clear();
    #endif  // TRITON_ENABLE_TRACING
    }

    // Returns whether all the responses from the state
    // are delivered and successfully written on the
    // stream.
    bool IsComplete() { return (complete_ && response_queue_->IsEmpty()); }

    void MarkAsAsyncNotifyState() { async_notify_state_ = true; }
    bool IsAsyncNotifyState() { return async_notify_state_; }
    // Needed in the response handle for classification outputs.
    TRITONSERVER_Server* tritonserver_;

    // Unique ID for the state. Used only for debugging so will
    // always be 0 in non-debug builds.
    uint64_t unique_id_;

    std::shared_ptr<Context> context_;
    Steps step_;
    std::recursive_mutex step_mtx_;

    // Shared pointer to the inference request object. The lifetime of
    // inference request object is extended till all the responses from
    // the request are processed and the request is released.
    std::shared_ptr<TRITONSERVER_InferenceRequest> inference_request_;

    bool is_decoupled_ = false;
    StateParameters parameters_;

    uint8_t status_;
    std::atomic<uint32_t> cb_count_;
    bool complete_;

    RequestTypePtr request_;
    std::shared_ptr<InferResponseQueue<ResponseType>> response_queue_;

    // For testing and debugging
    int delay_response_ms_;
    int delay_complete_ms_;
    int delay_process_ms_;

    // For inference requests the allocator payload, unused for other
    // requests.
    AllocPayload<ResponseType> alloc_payload_;

    // The below pointer is only set when using this state object as a
    // wrapper over actual state when being sent to completion queue
    // using AsyncNotifyWhenDone function. Otherwise it is nullptr.
    InferHandlerState* state_ptr_;

    // Tracks whether this state object has been wrapped and send to
    // AsyncNotifyWhenDone() function as a tag.
    bool async_notify_state_;
    uint8_t msg_infer_type_ = 0;
    std::shared_ptr<Connection> conn_;
    struct bufferevent* bev_ = nullptr;
    KytensorCodec* codec_ = nullptr;
    uint32_t msg_id_ = 0;
    void* handler_ = nullptr;
};

class InferHandler {
public:
    using State =
        InferHandlerState<inference::ModelInferRequest, ModelInferRequestPtr, inference::ModelInferResponse>;
    using StateContext = typename State::Context;

    InferHandler(
        const std::string& name,
        const std::shared_ptr<TRITONSERVER_Server>& tritonserver,
        const std::shared_ptr<triton::server::SharedMemoryManager>& shm_manager,
        KytensorCodec* codec,
        std::unordered_map<struct bufferevent*, std::shared_ptr<Connection>>* conns);
    ~InferHandler();

    void OnModelInfer(const struct bufferevent* bev,
        const MessageHeader& header,
        const ModelInferRequestPtr& request,
        int64_t receive_time);

State* StateNew(
    TRITONSERVER_Server* tritonserver,
    const std::shared_ptr<StateContext>& context,
    Steps start_step = Steps::START);

void StateRelease(State* state);

private:
    static void InferResponseComplete(
        TRITONSERVER_InferenceResponse* iresponse, const uint32_t flags,
        void* userp);

    std::string name_;
    std::shared_ptr<TRITONSERVER_Server> tritonserver_;
    const struct bufferevent* bev_;
    KytensorCodec* codec_;
    std::unordered_map<struct bufferevent*, std::shared_ptr<Connection>>* conns_ptr_{nullptr};
    uint32_t msg_id_;
    bool is_async_ = false;
    std::shared_ptr<triton::server::SharedMemoryManager> shm_manager_;

    TRITONSERVER_ResponseAllocator* allocator_;

    std::mutex alloc_mu_;
    const size_t max_state_bucket_count_ = 8;
    std::vector<State*> state_bucket_;
    KytensorResponse response_;
    std::thread send_response_worker_;
    std::deque<KytensorMessagePtr> response_queue_;
    std::condition_variable condition_response_;
};

}}