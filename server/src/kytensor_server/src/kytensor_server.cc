#include "kytensor_server.h"
#include "log.h"
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <unistd.h>
#include <event2/thread.h>
#include <google/protobuf/util/json_util.h>

namespace kytensor { namespace server {
TRITONSERVER_Error*
KytensorServer::Create(
    const std::shared_ptr<TRITONSERVER_Server>& tritonserver,
    const std::shared_ptr<triton::server::SharedMemoryManager>& shm_manager,
    const Options& server_options, std::unique_ptr<KytensorServer>* server)
{
    try {
        server->reset(new KytensorServer(tritonserver, shm_manager, server_options));
    }
    catch (const std::invalid_argument& pe) {
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, pe.what());
    }
    return nullptr;
}


KytensorServer::KytensorServer(
    const std::shared_ptr<TRITONSERVER_Server>& tritonserver,
        const std::shared_ptr<triton::server::SharedMemoryManager>& shm_manager,
        const Options& server_options)
    : codec_(std::bind(&KytensorDispatcher::RecieveKytensorMessage, &dispatcher_, _1, _2, _3, _4)),
      dispatcher_(std::bind(&KytensorServer::OnUnknownMessage, this, _1, _2, _3)),
      tritonserver_(tritonserver), shm_manager_(shm_manager), server_addr_(SOCKET_PATH),
      server_options_(server_options)
{
    model_infer_handler_ = std::make_unique<InferHandler>(
        "model infer handler", tritonserver_, shm_manager_, &codec_, &connections_);

    stream_infer_handler_ = std::make_unique<StreamInferHandler>(
        "stream infer handler", tritonserver_, shm_manager_, &codec_, &connections_);

    //common handler
    dispatcher_.RegisterMessageCallback<inference::Empty>(
        std::bind(&KytensorServer::OnEmpty, this, _1, _2, _3, _4));
    dispatcher_.RegisterMessageCallback<inference::ServerLiveRequest>(
        std::bind(&KytensorServer::OnServerLive, this, _1, _2, _3, _4));
    dispatcher_.RegisterMessageCallback<inference::ServerReadyRequest>(
        std::bind(&KytensorServer::OnServerReady, this, _1, _2, _3, _4));
    dispatcher_.RegisterMessageCallback<inference::ModelReadyRequest>(
        std::bind(&KytensorServer::OnModelReady, this, _1, _2, _3, _4));
    dispatcher_.RegisterMessageCallback<inference::ServerMetadataRequest>(
        std::bind(&KytensorServer::OnServerMetadata, this, _1, _2, _3, _4));
    dispatcher_.RegisterMessageCallback<inference::ModelMetadataRequest>(
    std::bind(&KytensorServer::OnModelMetadata, this, _1, _2, _3, _4));
    dispatcher_.RegisterMessageCallback<inference::ModelConfigRequest>(
    std::bind(&KytensorServer::OnModelConfig, this, _1, _2, _3, _4));
    dispatcher_.RegisterMessageCallback<inference::ModelStatisticsRequest>(
        std::bind(&KytensorServer::OnModelStatistics, this, _1, _2, _3, _4));
    dispatcher_.RegisterMessageCallback<inference::RepositoryIndexRequest>(
        std::bind(&KytensorServer::OnRepositoryIndex, this, _1, _2, _3, _4));
    dispatcher_.RegisterMessageCallback<inference::RepositoryModelLoadRequest>(
    std::bind(&KytensorServer::OnRepositoryModelLoad, this, _1, _2, _3, _4));
    dispatcher_.RegisterMessageCallback<inference::RepositoryModelUnloadRequest>(
        std::bind(&KytensorServer::OnRepositoryModelUnload, this, _1, _2, _3, _4));

    //model infer handler
    dispatcher_.RegisterMessageCallback<inference::ModelInferRequest>(
        [this](const bufferevent* bev, const MessageHeader& header,
               const std::shared_ptr<inference::ModelInferRequest>& msg,
               int64_t receive_time) {

            if (header.msg_infer_type == MESSAGE_INFER_STREAM) {
                stream_infer_handler_->OnModelStreamInfer(bev, header, msg, receive_time);
            } else {
                model_infer_handler_->OnModelInfer(bev, header, msg, receive_time);
            }
        });

}

KytensorServer::~KytensorServer() {

    if (base_) {
        event_base_loopbreak(base_);
    }

    if (event_loop_thread_.joinable())  {
        event_loop_thread_.join();
    }

    event_free(listener_event_);
    close(listener_fd_);
    event_base_free(base_);
    unlink(SOCKET_PATH);
}

void KytensorServer::EventLoop() {
    base_ = event_base_new();
    if (!base_) {
        LOG_ERR("%s: Could not initialize libevent!\n", __func__);
        return;
    }

    // Create the Unix domain socket
    listener_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (listener_fd_ < 0) {
        LOG_ERR("%s: socket failed\n", __func__);
        return;
    }

    // Initialize the socket address structure
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);
    // Unlink the socket path if it already exists
    unlink(SOCKET_PATH);

    // Bind the socket
    if (bind(listener_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        LOG_ERR("%s: bind failed\n", __func__);
        close(listener_fd_);
        return;
    }

    //set permissions for the socket file to allow users to connect
    chmod(SOCKET_PATH, 0666);

    // Listen for connections
    if (listen(listener_fd_, 16) < 0) {
        LOG_ERR("%s: listen failed\n", __func__);
        close(listener_fd_);
        return;
    }

    evutil_make_socket_nonblocking(listener_fd_);

    listener_event_ = event_new(base_, listener_fd_, EV_READ|EV_PERSIST, OnConnect, this);
    if (!listener_event_) {
        LOG_ERR("%s: event_new failed\n", __func__);
        close(listener_fd_);
        return;
    }
    event_add(listener_event_, NULL);

    LOG_INF("%s: Starting Kytensor server, local listening ...\n", __func__);

    event_base_dispatch(base_);
    LOGF_INF("Kytensor server event loop exited\n");
}

void KytensorServer::EventLoopTcp()
{
    base_ = event_base_new();
    if (!base_) {
        LOG_ERR("%s: Could not initialize libevent!\n", __func__);
        return;
    }

    // Create TCP listening socket (0.0.0.0:5555)
    const uint16_t SERVER_PORT = 5555;
    listener_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (listener_fd_ < 0) {
        LOG_ERR("%s: socket failed\n", __func__);
        return;
    }

    int reuse = 1;
    setsockopt(listener_fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(SERVER_PORT);

    if (bind(listener_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        LOG_ERR("%s: bind failed\n", __func__);
        close(listener_fd_);
        return;
    }

    if (listen(listener_fd_, 16) < 0) {
        LOG_ERR("%s: listen failed\n", __func__);
        close(listener_fd_);
        return;
    }

    evutil_make_socket_nonblocking(listener_fd_);

    listener_event_ = event_new(base_, listener_fd_, EV_READ|EV_PERSIST, OnConnect, this);
    if (!listener_event_) {
        LOG_ERR("%s: event_new failed\n", __func__);
        close(listener_fd_);
        return;
    }
    event_add(listener_event_, NULL);

    LOG_INF("%s: Starting Kytensor server, listening on 0.0.0.0:%u ...\n", __func__, SERVER_PORT);
    event_base_dispatch(base_);
}

TRITONSERVER_Error* KytensorServer::Start()
{
    common_log_set_verbosity_thold(server_options_.log_verbose_);
    common_log_set_prefix(common_log_main(), true);
    common_log_set_timestamps(common_log_main(), true);
    LOGF_INF("start Kytensor inference service\n");
    evthread_use_pthreads();
    if (!event_loop_thread_.joinable()) {
        event_loop_thread_ = std::thread(&KytensorServer::EventLoop, this);
        return nullptr;
    }

    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_ALREADY_EXISTS, "Kytensor server is already running.");
}

TRITONSERVER_Error* KytensorServer::Stop()
{
    if (base_) {
        event_base_loopbreak(base_);
    }

    if (event_loop_thread_.joinable())  {
        event_loop_thread_.join();
    }
    LOG_INF("%s: Kytensor server stopped\n", __func__);
    return nullptr;
}

void KytensorServer::ReceiveMessageCallback(bufferevent *bev, void *arg)
{
    KytensorServer* server = static_cast<KytensorServer*>(arg);

    auto it = server->connections_.find(bev);
    if (it != server->connections_.end()) {
        server->codec_.ServerReceiveMessage(it->second, arg);
    } else {
        LOGF_WRN("ReceiveMessageCallback: bev not found in connections map\n");
    }
}

void KytensorServer::OnConnect(evutil_socket_t listener, short event, void *arg)
{
    (void)event;
    struct sockaddr_un addr;
    socklen_t slen = sizeof(addr);
    int new_fd = accept(listener, (struct sockaddr*)&addr, &slen);

    if (new_fd < 0) {
        int error_no = errno;
        if (error_no == EAGAIN || error_no == EWOULDBLOCK) {
            // No pending connections; this can happen with non-blocking sockets
            return;
        }

        LOG_ERR("%s: accept failed\n", __func__);

        return;
    }

    LOGF_DBG("New connection accepted\n");

    struct bufferevent* bev;
    KytensorServer* server = static_cast<KytensorServer*>(arg);
    bev = bufferevent_socket_new(server->base_, new_fd, BEV_OPT_CLOSE_ON_FREE | BEV_OPT_THREADSAFE);
    //connection for per client
    auto conn = std::make_shared<Connection>(bev);
    server->connections_[bev] = conn;

    bufferevent_setcb(bev, ReceiveMessageCallback, OnClientWrite, OnClientEvent, arg);
    //set read/write watermark
    bufferevent_setwatermark(bev, EV_READ, 18, 0);
    bufferevent_setwatermark(bev, EV_WRITE, 18, 0);
    bufferevent_enable(bev, EV_READ|EV_WRITE);
}

void KytensorServer::OnClientWrite(bufferevent *bev, void *arg)
{
    (void)bev;
    (void)arg;
}

void KytensorServer::OnClientEvent(bufferevent *bev, short events, void *arg)
{
    KytensorServer* server = static_cast<KytensorServer*>(arg);

    if (events & BEV_EVENT_ERROR) {
        LOG_ERR("%s: error from bufferevent\n", __func__);
    }

    if (events & (BEV_EVENT_EOF | BEV_EVENT_ERROR)) {
        LOGF_DBG("client disconnect\n");

        auto it = server->connections_.find(bev);
        if (it != server->connections_.end()) {
            it->second->set_closed();
            server->connections_.erase(it);
        }
    }
}

void KytensorServer::OnUnknownMessage(const bufferevent *bev, const MessagePtr &message, int64_t receive_time)
{
    (void)bev;
    (void)receive_time;
    LOG_INF("%s: Received unknown message type: %s\n", __func__, message->GetTypeName().c_str());
}

void KytensorServer::SendResponse(const bufferevent* bev, const MessageHeader& header, const google::protobuf::Message& message)
{
    auto it = connections_.find(const_cast<bufferevent *>(bev));
    if (it != connections_.end()) {
        codec_.ServerSendMessage(it->second, header, message);
    } else {
        LOGF_WRN("SendResponse: bev not found in connections map\n");
    }
}

void KytensorServer::OnEmpty(const struct bufferevent* bev, const MessageHeader& header, const EmptyPtr& answer, int64_t receive_time)
{
    (void)receive_time;
    MessageHeader write_header;
    write_header.msg_id = header.msg_id;
    write_header.status = header.status;
    LOGF_DBG("Received answer: %s, id: %d\n", answer->GetTypeName().c_str(), answer->id());

    SendResponse(bev, write_header, *answer);
}

void
KytensorServer::OnServerLive(
    const bufferevent* bev,
    const MessageHeader& header,
    const ServerLiveRequestPtr& request,
    int64_t receive_time)
{
    (void)receive_time;
    LOGF_DBG("Received answer: %s, msg id: %d\n", request->GetTypeName().c_str(), header.msg_id);

    MessageHeader write_header;
    write_header.msg_id = header.msg_id;

    ServerLiveResponsePtr response = std::make_shared<inference::ServerLiveResponse>();

    bool live = false;
    TRITONSERVER_Error* err =
        TRITONSERVER_ServerIsLive(tritonserver_.get(), &live);

    response->set_live((err == nullptr) && live);

    uint8_t code;
    if (err == nullptr) {
        code = 0;
    } else {
        code = TRITONSERVER_ErrorCode(err) + 1;
        TRITONSERVER_ErrorDelete(err);
    }
    write_header.status = code;
    SendResponse(bev, write_header, *response);
}

void
KytensorServer::OnServerReady(
    const bufferevent* bev, const MessageHeader& header,
    const ServerReadyRequestPtr& request, int64_t receive_time)
{
    bool ready = false;
    MessageHeader write_header;
    write_header.msg_id = header.msg_id;
    TRITONSERVER_Error* err =
        TRITONSERVER_ServerIsReady(tritonserver_.get(), &ready);

    ServerReadyResponsePtr response = std::make_shared<inference::ServerReadyResponse>();
    response->set_ready((err == nullptr) && ready);

    uint8_t err_code;
    if (err == nullptr) {
        err_code = 0;
    } else {
        err_code = TRITONSERVER_ErrorCode(err) + 1;
        TRITONSERVER_ErrorDelete(err);
    }
    write_header.status = err_code;
    SendResponse(bev, write_header, *response);
}

// void
// KytensorServer::OnHealthCheck(
//     const bufferevent* bev, const MessageHeader& header,
//     const HealthCheckRequestPtr& request, int64_t receive_time)
// {
//     bool live = false;
//     MessageHeader write_header;
//     write_header.msg_id = header.msg_id;

//     TRITONSERVER_Error* err =
//         TRITONSERVER_ServerIsReady(tritonserver_.get(), &live);
//     auto serving_status = grpc::health::v1::HealthCheckResponse_ServingStatus_UNKNOWN;
//     if (err == nullptr) {
//         serving_status = live ? grpc::health::v1::HealthCheckResponse_ServingStatus_SERVING
//                              : grpc::health::v1::HealthCheckResponse_ServingStatus_NOT_SERVING;
//     }

//     HealthCheckResponsePtr response = std::make_shared<grpc::health::v1::HealthCheckResponse>();
//     response->set_status(serving_status);

//     uint8_t err_code;
//     if (err == nullptr) {
//         err_code = 0;
//     } else {
//         err_code = TRITONSERVER_ErrorCode(err) + 1;
//         TRITONSERVER_ErrorDelete(err);
//     }
//     write_header.status = err_code;
//     codec_.SendMessage(bev, write_header, *response);
// }

void
KytensorServer::OnModelReady(
    const bufferevent* bev, const MessageHeader& header,
    const ModelReadyRequestPtr& request, int64_t receive_time)
{
    bool is_ready = false;
    int64_t requested_model_version;
    MessageHeader write_header;
    write_header.msg_id = header.msg_id;

    auto err = GetModelVersionFromString(request->version(), &requested_model_version);
    if (err == nullptr) {
        err = TRITONSERVER_ServerModelIsReady(
            tritonserver_.get(), request->name().c_str(), requested_model_version, &is_ready);
    }

    ModelReadyResponsePtr response = std::make_shared<inference::ModelReadyResponse>();
    response->set_ready((err == nullptr) && is_ready);

    uint8_t err_code;
    if (err == nullptr) {
        err_code = 0;
    } else {
        err_code = TRITONSERVER_ErrorCode(err) + 1;
        TRITONSERVER_ErrorDelete(err);
    }
    write_header.status = err_code;
    SendResponse(bev, write_header, *response);
}

void
KytensorServer::OnServerMetadata(
    const bufferevent* bev, const MessageHeader& header,
    const ServerMetadataRequestPtr& request, int64_t receive_time)
{
    TRITONSERVER_Message* server_metadata_message = nullptr;
    ServerMetadataResponsePtr response = std::make_shared<inference::ServerMetadataResponse>();
    auto err = TRITONSERVER_ServerMetadata(tritonserver_.get(), &server_metadata_message);
    GOTO_IF_ERR(err, earlyexit);

    MessageHeader write_header;
    write_header.msg_id = header.msg_id;

    const char* buffer;
    size_t byte_size;
    err = TRITONSERVER_MessageSerializeToJson(
        server_metadata_message, &buffer, &byte_size);
    GOTO_IF_ERR(err, earlyexit);

    {
        triton::common::TritonJson::Value server_metadata_json;
        err = server_metadata_json.Parse(buffer, byte_size);
        GOTO_IF_ERR(err, earlyexit);

        const char* name;
        size_t namelen;
        err = server_metadata_json.MemberAsString("name", &name, &namelen);
        GOTO_IF_ERR(err, earlyexit);

        const char* version;
        size_t versionlen;
        err = server_metadata_json.MemberAsString("version", &version, &versionlen);
        GOTO_IF_ERR(err, earlyexit);

        response->set_name(std::string(name, namelen));
        response->set_version(std::string(version, versionlen));

        if(server_metadata_json.Find("extensions")) {
            triton::common::TritonJson::Value extensions_json;
            err = server_metadata_json.MemberAsArray("extensions", &extensions_json);
            GOTO_IF_ERR(err, earlyexit);

            for (size_t idx = 0; idx < extensions_json.ArraySize(); ++idx ) {
                const char* ext;
                size_t extlen;
                err = extensions_json.IndexAsString(idx, &ext, &extlen);
                GOTO_IF_ERR(err, earlyexit);
                response->add_extensions(std::string(ext, extlen));
            }
        }
        TRITONSERVER_MessageDelete(server_metadata_message);
    }

earlyexit:
    uint8_t err_code;
    if (err == nullptr) {
        err_code = 0;
    } else {
        err_code = TRITONSERVER_ErrorCode(err) + 1;
        TRITONSERVER_ErrorDelete(err);
    }
    write_header.status = err_code;
    SendResponse(bev, write_header, *response);
}

void
KytensorServer::OnModelMetadata(
    const bufferevent* bev, const MessageHeader& header,
    const ModelMetadataRequestPtr& request, int64_t receive_time)
{
    ModelMetadataResponsePtr response = std::make_shared<inference::ModelMetadataResponse>();
    int64_t requested_model_version;
    auto err = GetModelVersionFromString(request->version(), &requested_model_version);
    GOTO_IF_ERR(err, earlyexit);

    MessageHeader write_header;
    write_header.msg_id = header.msg_id;

    {
        TRITONSERVER_Message* model_metadata_message = nullptr;
        err = TRITONSERVER_ServerModelMetadata(
            tritonserver_.get(), request->name().c_str(), requested_model_version,
            &model_metadata_message);
        GOTO_IF_ERR(err, earlyexit);

        const char* buffer;
        size_t byte_size;
        err = TRITONSERVER_MessageSerializeToJson(
            model_metadata_message, &buffer, &byte_size);
        GOTO_IF_ERR(err, earlyexit);

        triton::common::TritonJson::Value model_metadata_json;
        err = model_metadata_json.Parse(buffer, byte_size);
        GOTO_IF_ERR(err, earlyexit);

        const char* name;
        size_t namelen;
        err = model_metadata_json.MemberAsString("name", &name, &namelen);
        GOTO_IF_ERR(err, earlyexit);

        response->set_name(std::string(name, namelen));

        if (model_metadata_json.Find("versions")) {
          triton::common::TritonJson::Value versions_json;
          err = model_metadata_json.MemberAsArray("versions", &versions_json);
          GOTO_IF_ERR(err, earlyexit);

          for (size_t idx = 0; idx < versions_json.ArraySize(); ++idx) {
            const char* version;
            size_t versionlen;
            err = versions_json.IndexAsString(idx, &version, &versionlen);
            GOTO_IF_ERR(err, earlyexit);
            response->add_versions(std::string(version, versionlen));
          }
        }

        const char* platform;
        size_t platformlen;
        err = model_metadata_json.MemberAsString(
            "platform", &platform, &platformlen);
        GOTO_IF_ERR(err, earlyexit);
        response->set_platform(std::string(platform, platformlen));

        if (model_metadata_json.Find("inputs")) {
          triton::common::TritonJson::Value inputs_json;
          err = model_metadata_json.MemberAsArray("inputs", &inputs_json);
          GOTO_IF_ERR(err, earlyexit);

          for (size_t idx = 0; idx < inputs_json.ArraySize(); ++idx) {
            triton::common::TritonJson::Value io_json;
            err = inputs_json.IndexAsObject(idx, &io_json);
            GOTO_IF_ERR(err, earlyexit);

            inference::ModelMetadataResponse::TensorMetadata* io =
                response->add_inputs();

            const char* name;
            size_t namelen;
            err = io_json.MemberAsString("name", &name, &namelen);
            GOTO_IF_ERR(err, earlyexit);

            const char* datatype;
            size_t datatypelen;
            err = io_json.MemberAsString("datatype", &datatype, &datatypelen);
            GOTO_IF_ERR(err, earlyexit);

            io->set_name(std::string(name, namelen));
            io->set_datatype(std::string(datatype, datatypelen));

            if (io_json.Find("shape")) {
              triton::common::TritonJson::Value shape_json;
              err = io_json.MemberAsArray("shape", &shape_json);
              GOTO_IF_ERR(err, earlyexit);

              for (size_t sidx = 0; sidx < shape_json.ArraySize(); ++sidx) {
                int64_t d;
                err = shape_json.IndexAsInt(sidx, &d);
                GOTO_IF_ERR(err, earlyexit);

                io->add_shape(d);
              }
            }
          }
        }

        if (model_metadata_json.Find("outputs")) {
          triton::common::TritonJson::Value outputs_json;
          err = model_metadata_json.MemberAsArray("outputs", &outputs_json);
          GOTO_IF_ERR(err, earlyexit);

          for (size_t idx = 0; idx < outputs_json.ArraySize(); ++idx) {
            triton::common::TritonJson::Value io_json;
            err = outputs_json.IndexAsObject(idx, &io_json);
            GOTO_IF_ERR(err, earlyexit);

            inference::ModelMetadataResponse::TensorMetadata* io =
                response->add_outputs();

            const char* name;
            size_t namelen;
            err = io_json.MemberAsString("name", &name, &namelen);
            GOTO_IF_ERR(err, earlyexit);

            const char* datatype;
            size_t datatypelen;
            err = io_json.MemberAsString("datatype", &datatype, &datatypelen);
            GOTO_IF_ERR(err, earlyexit);

            io->set_name(std::string(name, namelen));
            io->set_datatype(std::string(datatype, datatypelen));

            if (io_json.Find("shape")) {
              triton::common::TritonJson::Value shape_json;
              err = io_json.MemberAsArray("shape", &shape_json);
              GOTO_IF_ERR(err, earlyexit);

              for (size_t sidx = 0; sidx < shape_json.ArraySize(); ++sidx) {
                int64_t d;
                err = shape_json.IndexAsInt(sidx, &d);
                GOTO_IF_ERR(err, earlyexit);

                io->add_shape(d);
              }
            }
          }
        }

        TRITONSERVER_MessageDelete(model_metadata_message);
    }

earlyexit:
    uint8_t err_code;
    if (err == nullptr) {
        err_code = 0;
    } else {
        err_code = TRITONSERVER_ErrorCode(err) + 1;
        TRITONSERVER_ErrorDelete(err);
    }
    write_header.status = err_code;
    SendResponse(bev, write_header, *response);
}

void
KytensorServer::OnModelConfig(
    const bufferevent* bev, const MessageHeader& header,
    const ModelConfigRequestPtr& request, int64_t receive_time)
{
    ModelConfigResponsePtr response = std::make_shared<inference::ModelConfigResponse>();
    int64_t requested_model_version;
    MessageHeader write_header;
    write_header.msg_id = header.msg_id;

    auto err =
        GetModelVersionFromString(request->version(), &requested_model_version);
    if (err == nullptr) {
        TRITONSERVER_Message* model_config_message = nullptr;
        err = TRITONSERVER_ServerModelConfig(
            tritonserver_.get(), request->name().c_str(), requested_model_version,
            1 /* config_version */, &model_config_message);
        if (err == nullptr) {
            const char* buffer;
            size_t byte_size;
            err = TRITONSERVER_MessageSerializeToJson(
                model_config_message, &buffer, &byte_size);
            if (err == nullptr) {
            ::google::protobuf::util::JsonStringToMessage(
                ::google::protobuf::stringpiece_internal::StringPiece(
                    buffer, (int)byte_size),
                response->mutable_config());
            }
            TRITONSERVER_MessageDelete(model_config_message);
        }
    }

    uint8_t err_code;
    if (err == nullptr) {
        err_code = 0;
    } else {
        err_code = TRITONSERVER_ErrorCode(err) + 1;
        TRITONSERVER_ErrorDelete(err);
    }
    write_header.status = err_code;
    SendResponse(bev, write_header, *response);
}

void
KytensorServer::OnModelStatistics(
    const bufferevent* bev, const MessageHeader& header,
    const ModelStatisticsRequestPtr& request, int64_t receive_time)
{
    ModelStatisticsResponsePtr response = std::make_shared<inference::ModelStatisticsResponse>();
    MessageHeader write_header;
    write_header.msg_id = header.msg_id;

#ifdef TRITON_ENABLE_STATS
    triton::common::TritonJson::Value model_stats_json;

    int64_t requested_model_version;
    auto err =
        GetModelVersionFromString(request.version(), &requested_model_version);
    GOTO_IF_ERR(err, earlyexit);

    {
      TRITONSERVER_Message* model_stats_message = nullptr;
      err = TRITONSERVER_ServerModelStatistics(
          tritonserver_.get(), request.name().c_str(), requested_model_version,
          &model_stats_message);
      GOTO_IF_ERR(err, earlyexit);

      const char* buffer;
      size_t byte_size;
      err = TRITONSERVER_MessageSerializeToJson(
          model_stats_message, &buffer, &byte_size);
      GOTO_IF_ERR(err, earlyexit);

      err = model_stats_json.Parse(buffer, byte_size);
      GOTO_IF_ERR(err, earlyexit);

      TRITONSERVER_MessageDelete(model_stats_message);
    }

    if (model_stats_json.Find("model_stats")) {
      triton::common::TritonJson::Value stats_json;
      err = model_stats_json.MemberAsArray("model_stats", &stats_json);
      GOTO_IF_ERR(err, earlyexit);

      for (size_t idx = 0; idx < stats_json.ArraySize(); ++idx) {
        triton::common::TritonJson::Value model_stat;
        err = stats_json.IndexAsObject(idx, &model_stat);
        GOTO_IF_ERR(err, earlyexit);

        auto statistics = response->add_model_stats();

        const char* name;
        size_t namelen;
        err = model_stat.MemberAsString("name", &name, &namelen);
        GOTO_IF_ERR(err, earlyexit);

        const char* version;
        size_t versionlen;
        err = model_stat.MemberAsString("version", &version, &versionlen);
        GOTO_IF_ERR(err, earlyexit);

        statistics->set_name(std::string(name, namelen));
        statistics->set_version(std::string(version, versionlen));

        uint64_t ucnt;
        err = model_stat.MemberAsUInt("last_inference", &ucnt);
        GOTO_IF_ERR(err, earlyexit);
        statistics->set_last_inference(ucnt);

        err = model_stat.MemberAsUInt("inference_count", &ucnt);
        GOTO_IF_ERR(err, earlyexit);
        statistics->set_inference_count(ucnt);

        err = model_stat.MemberAsUInt("execution_count", &ucnt);
        GOTO_IF_ERR(err, earlyexit);
        statistics->set_execution_count(ucnt);

        {
          triton::common::TritonJson::Value infer_stats_json;
          err = model_stat.MemberAsObject("inference_stats", &infer_stats_json);
          GOTO_IF_ERR(err, earlyexit);

          err = SetStatisticsDuration(
              infer_stats_json, "success",
              statistics->mutable_inference_stats()->mutable_success());
          GOTO_IF_ERR(err, earlyexit);
          err = SetStatisticsDuration(
              infer_stats_json, "fail",
              statistics->mutable_inference_stats()->mutable_fail());
          GOTO_IF_ERR(err, earlyexit);
          err = SetStatisticsDuration(
              infer_stats_json, "queue",
              statistics->mutable_inference_stats()->mutable_queue());
          GOTO_IF_ERR(err, earlyexit);
          err = SetStatisticsDuration(
              infer_stats_json, "compute_input",
              statistics->mutable_inference_stats()->mutable_compute_input());
          GOTO_IF_ERR(err, earlyexit);
          err = SetStatisticsDuration(
              infer_stats_json, "compute_infer",
              statistics->mutable_inference_stats()->mutable_compute_infer());
          GOTO_IF_ERR(err, earlyexit);
          err = SetStatisticsDuration(
              infer_stats_json, "compute_output",
              statistics->mutable_inference_stats()->mutable_compute_output());
          GOTO_IF_ERR(err, earlyexit);
          err = SetStatisticsDuration(
              infer_stats_json, "cache_hit",
              statistics->mutable_inference_stats()->mutable_cache_hit());
          GOTO_IF_ERR(err, earlyexit);
          err = SetStatisticsDuration(
              infer_stats_json, "cache_miss",
              statistics->mutable_inference_stats()->mutable_cache_miss());
          GOTO_IF_ERR(err, earlyexit);
        }

        {
          triton::common::TritonJson::Value responses_json;
          err = model_stat.MemberAsObject("response_stats", &responses_json);
          GOTO_IF_ERR(err, earlyexit);

          std::vector<std::string> keys;
          err = responses_json.Members(&keys);
          GOTO_IF_ERR(err, earlyexit);

          for (const auto& key : keys) {
            triton::common::TritonJson::Value res_json;
            err = responses_json.MemberAsObject(key.c_str(), &res_json);
            GOTO_IF_ERR(err, earlyexit);

            inference::InferResponseStatistics res;

            err = SetStatisticsDuration(
                res_json, "compute_infer", res.mutable_compute_infer());
            GOTO_IF_ERR(err, earlyexit);
            err = SetStatisticsDuration(
                res_json, "compute_output", res.mutable_compute_output());
            GOTO_IF_ERR(err, earlyexit);
            err = SetStatisticsDuration(
                res_json, "success", res.mutable_success());
            GOTO_IF_ERR(err, earlyexit);
            err = SetStatisticsDuration(res_json, "fail", res.mutable_fail());
            GOTO_IF_ERR(err, earlyexit);
            err = SetStatisticsDuration(
                res_json, "empty_response", res.mutable_empty_response());
            GOTO_IF_ERR(err, earlyexit);
            err =
                SetStatisticsDuration(res_json, "cancel", res.mutable_cancel());
            GOTO_IF_ERR(err, earlyexit);

            (*statistics->mutable_response_stats())[key] = std::move(res);
          }
        }

        {
          triton::common::TritonJson::Value batches_json;
          err = model_stat.MemberAsArray("batch_stats", &batches_json);
          GOTO_IF_ERR(err, earlyexit);

          for (size_t idx = 0; idx < batches_json.ArraySize(); ++idx) {
            triton::common::TritonJson::Value batch_stat;
            err = batches_json.IndexAsObject(idx, &batch_stat);
            GOTO_IF_ERR(err, earlyexit);

            auto batch_statistics = statistics->add_batch_stats();

            uint64_t ucnt;
            err = batch_stat.MemberAsUInt("batch_size", &ucnt);
            GOTO_IF_ERR(err, earlyexit);
            batch_statistics->set_batch_size(ucnt);

            err = SetStatisticsDuration(
                batch_stat, "compute_input",
                batch_statistics->mutable_compute_input());
            GOTO_IF_ERR(err, earlyexit);
            err = SetStatisticsDuration(
                batch_stat, "compute_infer",
                batch_statistics->mutable_compute_infer());
            GOTO_IF_ERR(err, earlyexit);
            err = SetStatisticsDuration(
                batch_stat, "compute_output",
                batch_statistics->mutable_compute_output());
            GOTO_IF_ERR(err, earlyexit);
          }
        }

        {
          triton::common::TritonJson::Value memory_usage_json;
          err = model_stat.MemberAsArray("memory_usage", &memory_usage_json);
          GOTO_IF_ERR(err, earlyexit);

          for (size_t idx = 0; idx < memory_usage_json.ArraySize(); ++idx) {
            triton::common::TritonJson::Value usage;
            err = memory_usage_json.IndexAsObject(idx, &usage);
            GOTO_IF_ERR(err, earlyexit);

            auto memory_usage = statistics->add_memory_usage();
            {
              const char* type;
              size_t type_len;
              err = usage.MemberAsString("type", &type, &type_len);
              GOTO_IF_ERR(err, earlyexit);
              memory_usage->set_type(std::string(type, type_len));
            }
            {
              int64_t id;
              err = usage.MemberAsInt("id", &id);
              GOTO_IF_ERR(err, earlyexit);
              memory_usage->set_id(id);
            }
            {
              uint64_t byte_size;
              err = usage.MemberAsUInt("byte_size", &byte_size);
              GOTO_IF_ERR(err, earlyexit);
              memory_usage->set_byte_size(byte_size);
            }
          }
        }
      }
    }

earlyexit:
    uint8_t err_code;
    if (err == nullptr) {
        err_code = 0;
    } else {
        err_code = TRITONSERVER_ErrorCode(err);
        TRITONSERVER_ErrorDelete(err);
    }
    codec_.SendMessage(bev, msg_id, err_code, *response);
#else
    auto err = TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNAVAILABLE,
        "the server does not support model statistics");
    uint8_t err_code;
    if (err == nullptr) {
        err_code = 0;
    } else {
        err_code = TRITONSERVER_ErrorCode(err) + 1;
        TRITONSERVER_ErrorDelete(err);
    }
    write_header.status = err_code;
    SendResponse(bev, write_header, *response);
#endif
}

void
KytensorServer::OnRepositoryIndex(
    const bufferevent* bev, const MessageHeader& header,
    const RepositoryIndexRequestPtr& request, int64_t receive_time)
{
    RepositoryIndexResponsePtr response = std::make_shared<inference::RepositoryIndexResponse>();
    TRITONSERVER_Error* err = nullptr;
    MessageHeader write_header;
    write_header.msg_id = header.msg_id;

    if (request->repository_name().empty()) {
        uint32_t flags = 0;
        if (request->ready()) {
            flags |= TRITONSERVER_INDEX_FLAG_READY;
        }

        TRITONSERVER_Message* model_index_message = nullptr;
        err = TRITONSERVER_ServerModelIndex(
            tritonserver_.get(), flags, &model_index_message);
        GOTO_IF_ERR(err, earlyexit);

        const char* buffer;
        size_t byte_size;
        err = TRITONSERVER_MessageSerializeToJson(
            model_index_message, &buffer, &byte_size);
        GOTO_IF_ERR(err, earlyexit);

        triton::common::TritonJson::Value model_index_json;
        err = model_index_json.Parse(buffer, byte_size);
        GOTO_IF_ERR(err, earlyexit);

        err = model_index_json.AssertType(
            triton::common::TritonJson::ValueType::ARRAY);
        GOTO_IF_ERR(err, earlyexit);

        for (size_t idx = 0; idx < model_index_json.ArraySize(); ++idx) {
        triton::common::TritonJson::Value index_json;
        err = model_index_json.IndexAsObject(idx, &index_json);
        GOTO_IF_ERR(err, earlyexit);

        auto model_index = response->add_models();

        const char* name;
        size_t namelen;
        err = index_json.MemberAsString("name", &name, &namelen);
        GOTO_IF_ERR(err, earlyexit);
        model_index->set_name(std::string(name, namelen));

        if (index_json.Find("version")) {
            const char* version;
            size_t versionlen;
            err = index_json.MemberAsString("version", &version, &versionlen);
            GOTO_IF_ERR(err, earlyexit);
            model_index->set_version(std::string(version, versionlen));
        }
        if (index_json.Find("state")) {
            const char* state;
            size_t statelen;
            err = index_json.MemberAsString("state", &state, &statelen);
            GOTO_IF_ERR(err, earlyexit);
            model_index->set_state(std::string(state, statelen));
        }
        if (index_json.Find("reason")) {
            const char* reason;
            size_t reasonlen;
            err = index_json.MemberAsString("reason", &reason, &reasonlen);
            GOTO_IF_ERR(err, earlyexit);
            model_index->set_reason(std::string(reason, reasonlen));
        }
        }

        TRITONSERVER_MessageDelete(model_index_message);
    } else {
        err = TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_UNSUPPORTED,
            "'repository_name' specification is not supported");
    }

earlyexit:
    uint8_t err_code;
    if (err == nullptr) {
        err_code = 0;
    } else {
        err_code = TRITONSERVER_ErrorCode(err) + 1;
        TRITONSERVER_ErrorDelete(err);
    }
    write_header.status = err_code;
    SendResponse(bev, write_header, *response);
}

void
KytensorServer::OnRepositoryModelLoad(
    const bufferevent* bev, const MessageHeader& header,
    const RepositoryModelLoadRequestPtr& request, int64_t receive_time)
{
    RepositoryModelLoadResponsePtr response = std::make_shared<inference::RepositoryModelLoadResponse>();
    TRITONSERVER_Error* err = nullptr;
    MessageHeader write_header;
    write_header.msg_id = header.msg_id;

    if (request->repository_name().empty()) {
        std::vector<TRITONSERVER_Parameter*> params;
        std::vector<const TRITONSERVER_Parameter*> const_params;
        for (const auto& param_proto : request->parameters()) {
            if (param_proto.first == "config") {
                if (param_proto.second.parameter_choice_case() !=
                    inference::ModelRepositoryParameter::ParameterChoiceCase::
                    kStringParam) {
                        err = TRITONSERVER_ErrorNew(
                            TRITONSERVER_ERROR_INVALID_ARG,
                            (std::string("invalid value type for load paramater '") +
                            param_proto.first + "', expected string_param.").c_str());
                        break;
                } else {
                    auto param = TRITONSERVER_ParameterNew(
                        param_proto.first.c_str(), TRITONSERVER_PARAMETER_STRING,
                        param_proto.second.string_param().c_str());
                    if (param != nullptr) {
                        params.emplace_back(param);
                        const_params.emplace_back(param);
                    } else {
                        err = TRITONSERVER_ErrorNew(
                            TRITONSERVER_ERROR_INTERNAL,
                            "unexpected error on creating Triton parameter");
                        break;
                    }
                }
            } else if (param_proto.first.rfind("file:", 0) == 0) {
                if (param_proto.second.parameter_choice_case() !=
                    inference::ModelRepositoryParameter::ParameterChoiceCase::
                        kBytesParam) {
                    err = TRITONSERVER_ErrorNew(
                        TRITONSERVER_ERROR_INVALID_ARG,
                        (std::string("invalid value type for load parameter '") +
                        param_proto.first + "', expected bytes_param.")
                            .c_str());
                    break;
                } else {
                    auto param = TRITONSERVER_ParameterBytesNew(
                        param_proto.first.c_str(),
                        param_proto.second.bytes_param().data(),
                        param_proto.second.bytes_param().length());
                    if (param != nullptr) {
                        params.emplace_back(param);
                        const_params.emplace_back(param);
                    } else {
                        err = TRITONSERVER_ErrorNew(
                            TRITONSERVER_ERROR_INTERNAL,
                            "unexpected error on creating Triton parameter");
                        break;
                    }
                    }
            } else {
                err = TRITONSERVER_ErrorNew(
                    TRITONSERVER_ERROR_INVALID_ARG,
                    (std::string("unrecognized load parameter '") +
                        param_proto.first + "'.")
                        .c_str());
                break;
            }
        }

        if (err == nullptr) {
            err = TRITONSERVER_ServerLoadModelWithParameters(
                tritonserver_.get(), request->model_name().c_str(),
                const_params.data(), const_params.size());
        }
        for (auto& param : params) {
            TRITONSERVER_ParameterDelete(param);
        }
    } else {
        err = TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_UNSUPPORTED,
            "'repository_name' specification is not supported");
    }

    uint8_t err_code;
    if (err == nullptr) {
        err_code = 0;
    } else {
        err_code = TRITONSERVER_ErrorCode(err) + 1;
        TRITONSERVER_ErrorDelete(err);
    }
    write_header.status = err_code;
    SendResponse(bev, write_header, *response);
}

void
KytensorServer::OnRepositoryModelUnload(
    const bufferevent* bev, const MessageHeader& header,
    const RepositoryModelUnloadRequestPtr& request, int64_t receive_time)
{
    RepositoryModelUnloadResponsePtr response = std::make_shared<inference::RepositoryModelUnloadResponse>();
    TRITONSERVER_Error* err = nullptr;
    MessageHeader write_header;
    write_header.msg_id = header.msg_id;
    if (request->repository_name().empty()) {
        // Check if the dependent models should be removed
        bool unload_dependents = false;
        for (auto param : request->parameters()) {
            if (param.first.compare("unload_dependents") == 0) {
                const auto& unload_param = param.second;
                if (unload_param.parameter_choice_case() !=
                    inference::ModelRepositoryParameter::ParameterChoiceCase::
                        kBoolParam) {
                err = TRITONSERVER_ErrorNew(
                    TRITONSERVER_ERROR_INVALID_ARG,
                    "invalid value type for 'unload_dependents' parameter, "
                    "expected "
                    "bool_param.");
                }
                unload_dependents = unload_param.bool_param();
                break;
            }
        }
        if (err == nullptr) {
            if (unload_dependents) {
            err = TRITONSERVER_ServerUnloadModelAndDependents(
                tritonserver_.get(), request->model_name().c_str());
            } else {
            err = TRITONSERVER_ServerUnloadModel(
                tritonserver_.get(), request->model_name().c_str());
            }
        }
    } else {
    err = TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "'repository_name' specification is not supported");
    }

    uint8_t err_code;
    if (err == nullptr) {
        err_code = 0;
    } else {
        err_code = TRITONSERVER_ErrorCode(err) + 1;
        TRITONSERVER_ErrorDelete(err);
    }
    write_header.status = err_code;
    SendResponse(bev, write_header, *response);
}

}}