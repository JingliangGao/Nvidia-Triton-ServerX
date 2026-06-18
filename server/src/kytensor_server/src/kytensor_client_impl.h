#pragma once

#include "kytensor_client.h"
#include "kytensor_codec.h"
#include "kytensor_dispatcher.h"
#include "kytensor_response.h"
#include "kytensor_common.h"

#include <event2/event.h>
#include <event2/bufferevent.h>
#include <sys/time.h>
#include <thread>
#include <future>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <unordered_map>

using std::placeholders::_1;
using std::placeholders::_2;
using std::placeholders::_3;
using std::placeholders::_4;
using std::placeholders::_5;

namespace kytensor { namespace client {

namespace ks = kytensor::server;

#define SOCKET_PATH "/tmp/kytensor_server_socket"

class KyTensorImpl {
public:

    static Error Create(std::unique_ptr<KyTensorImpl>* client, bool verbose = false);

    KyTensorImpl(bool verbose);
    ~KyTensorImpl();

    bool Start(bool verbose);
    void Stop();
    void SyncSendMessage(const std::string& content);
    void AsyncSendMessage(KyTensorClient::OnCompleteFunc callback, const int id);

    Error IsServerLive(bool* live, const uint64_t timeout_ms = 0);
    Error IsServerReady(bool* ready, const uint64_t timeout_ms = 0);
    Error IsModelReady(bool* ready, const std::string& model_name,
                     const std::string& model_version = "", const uint64_t timeout_ms = 0);
    Error ServerMetadata(inference::ServerMetadataResponse* server_metadata,
                       const uint64_t timeout_ms = 0);
    Error ModelMetadata(inference::ModelMetadataResponse* model_metadata,
                      const std::string& model_name, const std::string& model_version = "",
                      const uint64_t timeout_ms = 0);
    Error ModelConfig(inference::ModelConfigResponse* model_config,
                    const std::string& model_name, const std::string& model_version = "",
                    const uint64_t timeout_ms = 0);
    Error ModelRepositoryIndex(inference::RepositoryIndexResponse* repository_index,
                           const uint64_t timeout_ms = 0);
    Error LoadModel(const std::string& model_name,
                  const std::string& config = std::string(),
                  const std::map<std::string, std::vector<char>>& files = {},
                  const uint64_t timeout_ms = 0);
    Error UnloadModel(const std::string& model_name, const uint64_t timeout_ms = 0);
    Error ModelInferenceStatistics(inference::ModelStatisticsResponse* infer_stat,
                                const std::string& model_name = "", const std::string& model_version = "",
                                const uint64_t timeout_ms = 0);

    Error Infer(InferResult** result, const InferOptions& options,
               const std::vector<InferInput*>& inputs,
               const std::vector<const InferRequestedOutput*>& outputs,
               KYTENSOR_COMPRESSION_TYPE compression_type);

    Error AsyncInfer(KyTensorClient::OnCompleteFn callback, const InferOptions& options,
                    const std::vector<InferInput*>& inputs,
                    const std::vector<const InferRequestedOutput*>& outputs,
                    KYTENSOR_COMPRESSION_TYPE compression_type);

    Error StartStream(KyTensorClient::OnCompleteFn callback, bool enable_stats,
                     uint32_t stream_timeout, KYTENSOR_COMPRESSION_TYPE compression_type);

    Error StopStream();

    Error AsyncStreamInfer(const InferOptions& options, const std::vector<InferInput*>& inputs,
                         const std::vector<const InferRequestedOutput*>& outputs);

    std::promise<void> started_promise_;
    bool is_started_ = false;

private:
    static void ReceiveMessageCallback(struct bufferevent *bev, void *arg);
    static void OnWrite(bufferevent *bev, void *arg);
    static void OnConnect(struct bufferevent *bev, short events, void *arg);

    void OnUnknownMessage(const struct bufferevent* bev,
                          const ks::MessagePtr& message,
                          int64_t receive_time);

    void OnEmpty(const struct bufferevent* bev,
                 const ks::MessageHeader& header,
                 const EmptyPtr& msg,
                 int64_t receive_time);

    void OnStreamInferResponse(const struct bufferevent* bev,
                            const ks::MessageHeader& header,
                            const std::shared_ptr<inference::ModelStreamInferResponse>& msg,
                            int64_t receive_time);

    template<typename T>
    void HandleResponse(const bufferevent* bev,
                      const ks::MessageHeader& header,
                      const std::shared_ptr<T> msg,
                      int64_t receive_time);

    void EventLoop();
    void EventLoopTcp();

    Error PreRunProcessing(const InferOptions& options, const std::vector<InferInput*>& inputs,
                         const std::vector<const InferRequestedOutput*>& outputs);

    int msg_id_ = 0;

    struct event_base* base_;
    struct bufferevent* bev_;
    ks::KytensorCodec codec_;
    ks::KytensorDispatcher dispatcher_;
    ks::KytensorResponse response_;
    uint64_t send_time_;

    std::thread event_loop_thread_;
    KyTensorClient::OnCompleteFunc callback_;
    std::unordered_map<int, KyTensorClient::OnCompleteFn> infer_callbacks_;
    bool running_ = true;

    std::mutex mutex_request_;

    inference::ModelInferRequest infer_request_;

    int async_msg_id_ = 0;
    int async_recv_num_ = 0;
    int async_last_distance_ = 0;
    std::mutex mutex_async_request_;

    KyTensorClient::OnCompleteFn stream_callback_;
    bool stream_running_;
    int stream_msg_id_ = 0;
    int stream_recv_num_ = 0;
    int stream_last_distance_ = 0;
    std::mutex mutex_stream_request_;

    const double kalpha_ = 0.25;
    const int ktarget_distance_ = 25;
    const int kdelay_unit_us_ = 20;
    int kmax_delay_us_ = 0;

    std::shared_ptr<std::promise<bool>> start_promise_;
};

}}