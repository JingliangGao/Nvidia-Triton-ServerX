#pragma once
#include <vector>
#include <thread>
#include <functional>
#include <unordered_map>
#include <event2/event.h>
#include <event2/bufferevent.h>

#include "kytensor_codec.h"
#include "kytensor_dispatcher.h"
#include "infer_handler.h"
#include "stream_infer_handler.h"

#include "../../restricted_features.h"
#include "../../shared_memory_manager.h"

using std::placeholders::_1;
using std::placeholders::_2;
using std::placeholders::_3;
using std::placeholders::_4;
using std::placeholders::_5;

namespace kytensor { namespace server {

#define SOCKET_PATH "/tmp/kytensor_server_socket"

struct SocketOptions {
    std::string address_{"0.0.0.0"};
    int32_t port_{8001};
    bool reuse_port_{false};
};

struct Options {
    SocketOptions socket_;
    int32_t log_verbose_{0};
};

class KytensorServer {

public:
    static TRITONSERVER_Error* Create(
        const std::shared_ptr<TRITONSERVER_Server>& tritonserver,
        const std::shared_ptr<triton::server::SharedMemoryManager>& shm_manager,
        const Options& server_options, std::unique_ptr<KytensorServer>* server);

    KytensorServer(
        const std::shared_ptr<TRITONSERVER_Server>& tritonserver,
        const std::shared_ptr<triton::server::SharedMemoryManager>& shm_manager,
        const Options& server_options);
    ~KytensorServer();

    TRITONSERVER_Error* Start();
    TRITONSERVER_Error* Stop();

    std::unordered_map<struct bufferevent*, std::shared_ptr<Connection>> connections_;
private:

    static void ReceiveMessageCallback(struct bufferevent *bev, void *arg);
    static void OnConnect(evutil_socket_t listener, short event, void *arg);
    static void OnClientWrite(struct bufferevent *bev, void *arg);
    static void OnClientEvent(struct bufferevent *bev, short events, void *arg);
    void SendResponse(const bufferevent* bev, const MessageHeader& header, const google::protobuf::Message& message);
    void OnUnknownMessage(const struct bufferevent* bev,
                          const MessagePtr& message,
                          int64_t receive_time);
    void OnEmpty(const struct bufferevent* bev,
                const MessageHeader& header,
                const EmptyPtr& answer,
                int64_t receive_time);
    void OnServerLive(const struct bufferevent* bev,
                    const MessageHeader& header,
                    const ServerLiveRequestPtr& request,
                    int64_t receive_time);
    void OnServerReady(const struct bufferevent* bev,
        const MessageHeader& header,
        const ServerReadyRequestPtr& request,
        int64_t receive_time);
    // void OnHealthCheck(const struct bufferevent* bev,
    //     const MessageHeader& header,
    //     const HealthCheckRequestPtr& request,
    //     int64_t receive_time);
    void OnModelReady(const struct bufferevent* bev,
        const MessageHeader& header,
        const ModelReadyRequestPtr& request,
        int64_t receive_time);
    void OnServerMetadata(const struct bufferevent* bev,
        const MessageHeader& header,
        const ServerMetadataRequestPtr& request,
        int64_t receive_time);
    void OnModelMetadata(const struct bufferevent* bev,
        const MessageHeader& header,
        const ModelMetadataRequestPtr& request,
        int64_t receive_time);
    void OnModelConfig(const struct bufferevent* bev,
        const MessageHeader& header,
        const ModelConfigRequestPtr& request,
        int64_t receive_time);
    void OnModelStatistics(const struct bufferevent* bev,
        const MessageHeader& header,
        const ModelStatisticsRequestPtr& request,
        int64_t receive_time);
    void OnRepositoryIndex(const struct bufferevent* bev,
        const MessageHeader& header,
        const RepositoryIndexRequestPtr& request,
        int64_t receive_time);
    void OnRepositoryModelLoad(const struct bufferevent* bev,
        const MessageHeader& header,
        const RepositoryModelLoadRequestPtr& request,
        int64_t receive_time);
    void OnRepositoryModelUnload(const struct bufferevent* bev,
        const MessageHeader& header,
        const RepositoryModelUnloadRequestPtr& request,
        int64_t receive_time);

    void EventLoop();
    void EventLoopTcp();

    struct event_base* base_;
    int listener_fd_;
    struct event* listener_event_;

    KytensorCodec codec_;
    KytensorDispatcher dispatcher_;

    std::thread event_loop_thread_;

    std::shared_ptr<TRITONSERVER_Server> tritonserver_;
    std::shared_ptr<triton::server::SharedMemoryManager> shm_manager_;
    const std::string server_addr_;
    Options server_options_;

    std::unique_ptr<InferHandler> model_infer_handler_;
    std::unique_ptr<StreamInferHandler> stream_infer_handler_;
};

}}