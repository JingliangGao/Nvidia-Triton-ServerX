#include "kytensor_response.h"
#include "log.h"

namespace kytensor { namespace server {

void KytensorResponse::AddResponseId(uint32_t id)
{
    LOGF_DBG("add response id = %d to queue, current ids size %ld (before add)\n", id, response_ids_.size());

    std::unique_lock<std::mutex> lock(mutex_response_);
    response_ids_.insert(id);
}

void KytensorResponse::RemoveResponseId(uint32_t id) {
    LOGF_DBG("remove response id = %d to queue, current ids size %ld (before remove)\n", id, response_ids_.size());

    std::unique_lock<std::mutex> lock(mutex_response_);
    response_ids_.erase(id);
    response_queue_.erase(
        std::remove_if(response_queue_.begin(), response_queue_.end(),
                       [id](const KytensorMessagePtr& msg) { return msg->msg_id == id; }),
        response_queue_.end());
}

void KytensorResponse::SendResponse(KytensorMessagePtr&& msg) {
    LOGF_DBG("sending response for id = %d\n", msg->msg_id);

    std::unique_lock<std::mutex> lock(mutex_response_);
    for (auto & id : response_ids_) {
        if(id == (int)msg->msg_id) {
            response_queue_.emplace_back(std::move(msg));
            condition_response_.notify_all();
            return;
        }
    }
}

void KytensorResponse::SendResponseWithoutId(KytensorMessagePtr&& msg) {
    LOGF_INF("sending response for id = %d\n", msg->msg_id);

    std::unique_lock<std::mutex> lock(mutex_response_);

    response_queue_.emplace_back(std::move(msg));
    condition_response_.notify_all();

}

KytensorMessagePtr KytensorResponse::RecvResponse(uint32_t id) {
    while (true) {
        std::unique_lock<std::mutex> lock(mutex_response_);
        condition_response_.wait(lock, [&]{
            return !response_queue_.empty();
        });

        for (size_t i = 0; i < response_queue_.size(); i++) {
            if (response_queue_[i]->msg_id == id) {
                KytensorMessagePtr res = std::move(response_queue_[i]);
                response_queue_.erase(response_queue_.begin() + i);
                return res;
            }
        }
    }
}

KytensorMessagePtr KytensorResponse::RecvResponse(uint32_t id, int timeout) {
    while (true) {
        std::unique_lock<std::mutex> lock(mutex_response_);

        for (size_t i = 0; i < response_queue_.size(); i++) {
            if (response_queue_[i]->msg_id == id) {
                KytensorMessagePtr res = std::move(response_queue_[i]);
                response_queue_.erase(response_queue_.begin() + i);
                return res;
            }
        }

        std::cv_status cr_res = condition_response_.wait_for(lock, std::chrono::milliseconds(timeout));
        if (cr_res == std::cv_status::timeout) {
            return nullptr;
        }
    }
}

KytensorMessagePtr KytensorResponse::RecvResponse()
{
    while (true) {
        std::unique_lock<std::mutex> lock(mutex_response_);
        condition_response_.wait(lock, [&]{
            return !response_queue_.empty();
        });
        LOGF_INF("response_queue size: %ld\n", response_queue_.size());
        KytensorMessagePtr res = std::move(response_queue_[0]);
        response_queue_.erase(response_queue_.begin());
        return res;
    }
}

KytensorMessagePtr KytensorResponse::TryRecvResponse()
{
    KytensorMessagePtr res = std::move(response_queue_[0]);
    response_queue_.erase(response_queue_.begin());
    return res;
}

}}