#pragma once
#include <list>
#include <unordered_map>

#include "triton/core/tritonserver.h"
#include "log.h"

namespace kytensor { namespace server {

//
// ShmInfo
//
// Simple structure that carries the shared memory information
//
struct ShmInfo {
    ShmInfo(
        void* base, size_t byte_size, TRITONSERVER_MemoryType memory_type,
        int64_t memory_type_id, char* cuda_ipc_handle)
        : base_(base), byte_size_(byte_size), memory_type_(memory_type),
            memory_type_id_(memory_type_id), cuda_ipc_handle_(cuda_ipc_handle)
    {
    }
    void* base_;
    size_t byte_size_;
    TRITONSERVER_MemoryType memory_type_;
    int64_t memory_type_id_;
    char* cuda_ipc_handle_;
};

using TensorShmMap = std::unordered_map<std::string, ShmInfo>;

template <typename ResponseType>
class InferResponseQueue {
    public:
    explicit InferResponseQueue() { Reset(); }

    ~InferResponseQueue()
    {
        for (auto response : responses_) {
            delete response;
        }
    }

    // Resets the queue
    void Reset()
    {
        alloc_count_ = 0;
        ready_count_ = 0;
        current_index_ = 0;
        for (auto response : responses_) {
            response->Clear();
        }
    }

    // Gets the response for the non-decoupled models.
    // Note that there will be a single response in
    // non-decoupled cases.
    ResponseType* GetNonDecoupledResponse()
    {
        std::lock_guard<std::mutex> lock(mtx_);
        alloc_count_ = 1;
        if (responses_.size() < 1) {
            responses_.push_back(new ResponseType());
        }
        return responses_[0];
    }

    // Allocates a response on the head of the queue
    void AllocateResponse()
    {
        std::lock_guard<std::mutex> lock(mtx_);
        alloc_count_++;
        if (responses_.size() < alloc_count_) {
            responses_.push_back(new ResponseType());
        }
    }

    // Gets the last allocated response
    ResponseType* GetLastAllocatedResponse()
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (responses_.size() < alloc_count_) {
            LOGF_ERR("[INTERNAL] Attempting to access the response not yet allocated");
            return nullptr;
        }
        return responses_[alloc_count_ - 1];
    }

    // Marks the next non-ready response complete
    bool MarkNextResponseComplete()
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (alloc_count_ <= ready_count_) {
            LOGF_ERR("[INTERNAL] Attempting to mark an unallocated response complete");
            return false;
        }
        ready_count_++;

        return true;
    }

    // Gets the current response from the tail of
    // the queue.
    ResponseType* GetCurrentResponse()
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (current_index_ >= ready_count_) {
            LOGF_ERR("[INTERNAL] Attempting to access current response when it ");
            return nullptr;
        }
        return responses_[current_index_];
    }

    // Gets the response at the specified index
    ResponseType* GetResponseAt(const uint32_t index)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (index >= alloc_count_) {
            LOGF_ERR("[INTERNAL] Attempting to access response which is not yet allocated");
            return nullptr;
        }
        return responses_[index];
    }

    // Pops the response from the tail of the queue
    void PopResponse()
    {
        std::lock_guard<std::mutex> lock(mtx_);
        current_index_++;
    }

    // Returns whether the queue is empty
    bool IsEmpty()
    {
        std::lock_guard<std::mutex> lock(mtx_);
        return ((alloc_count_ == ready_count_) && (alloc_count_ == current_index_));
    }

    // Returns whether the queue has responses
    // ready to be written.
    bool HasReadyResponse()
    {
        std::lock_guard<std::mutex> lock(mtx_);
        return (ready_count_ > current_index_);
    }

private:
    std::vector<ResponseType*> responses_;
    std::mutex mtx_;

    // There are three indices to track the responses in the queue
    // Tracks the allocated response
    uint32_t alloc_count_;
    // Tracks the response that is ready to be written
    uint32_t ready_count_;
    // Tracks the response next in the queue to be written
    uint32_t current_index_;
};

//
// AllocPayload
//
// Simple structure that carries the userp payload needed for
// allocation.
//
template <typename ResponseType>
struct AllocPayload {
    using ClassificationMap = std::unordered_map<std::string, uint32_t>;

    explicit AllocPayload() : response_queue_(nullptr) {}
    ~AllocPayload()
    {
        // Don't delete 'response_'.. it is owned by the InferHandlerState
    }

    std::shared_ptr<InferResponseQueue<ResponseType>> response_queue_;
    uint32_t response_alloc_count_;
    TensorShmMap shm_map_;
    ClassificationMap classification_map_;

    // Used to extend the lifetime of the serialized data in case
    // non-raw contents were provided in the request. Serialized data's
    // actual lifetime is that of the request whereas AllocPayload's
    // lifetime is that of a response... but it is convenient to keep it
    // here.
    std::list<std::string> serialized_data_;
};

}}