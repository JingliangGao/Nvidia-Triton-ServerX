#pragma once

#include <mutex>
#include <condition_variable>
#include <unordered_set>

#include "kytensor_service.pb.h"
#include "kytensor_common.h"

namespace kytensor { namespace server {

struct KytensorMessage {
    uint32_t msg_id;
    uint8_t status;
    MessagePtr message;
};

using KytensorMessagePtr = std::unique_ptr<KytensorMessage>;

class KytensorResponse {
public:
    KytensorResponse() {};
    ~KytensorResponse() {};

    void AddResponseId(uint32_t id);
    void RemoveResponseId(uint32_t id);
    void SendResponse(KytensorMessagePtr&& msg);
    void SendResponseWithoutId(KytensorMessagePtr&& msg);
    KytensorMessagePtr RecvResponse(uint32_t id);
    KytensorMessagePtr RecvResponse(uint32_t id, int timeout);
    KytensorMessagePtr RecvResponse();
    KytensorMessagePtr TryRecvResponse();

private:

    std::unordered_set<int> response_ids_;
    std::mutex mutex_response_;
    std::vector<KytensorMessagePtr> response_queue_;
    std::condition_variable condition_response_;
};

}}