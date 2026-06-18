#define TRITON_INFERENCE_SERVER_CLIENT_CLASS KyTensorImpl
#include "kytensor_client_impl.h"
#include "kytensor_response.h"
#include "log.h"
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <event2/thread.h>
#include <event2/buffer.h>
#include <cmath>

namespace kytensor { namespace client {

static int64_t t_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

template<typename T>
void KyTensorImpl::HandleResponse(
    const bufferevent* bev,
    const ks::MessageHeader& header,
    const std::shared_ptr<T> msg,
    int64_t receive_time)
{
    LOG_DBG("%s: Received message: %s, msg id: %d, status: %d\n",
        __func__, msg->GetTypeName().c_str(), header.msg_id, header.status);

    auto response = std::make_unique<ks::KytensorMessage>();
    response->msg_id = header.msg_id;
    response->status = header.status;
    response->message = msg;
    response_.SendResponse(std::move(response));
}

class InferResultKyTensor : public InferResult {
public:
    static Error Create(
         InferResult** infer_result,
         std::shared_ptr<inference::ModelInferResponse> response,
         Error& request_status);
    static Error Create(
         InferResult** infer_result,
         std::shared_ptr<inference::ModelStreamInferResponse> response);

    Error RequestStatus() const override;
    Error ModelName(std::string* name) const override;
    Error ModelVersion(std::string* version) const override;
    Error Id(std::string* id) const override;
    Error Shape(const std::string& output_name, std::vector<int64_t>* shape)
        const override;
    Error Datatype(
        const std::string& output_name, std::string* datatype) const override;
    Error RawData(
        const std::string& output_name, const uint8_t** buf,
        size_t* byte_size) const override;
    Error IsFinalResponse(bool* is_final_response) const override;
    Error IsNullResponse(bool* is_null_response) const override;
    Error StringData(
        const std::string& output_name,
        std::vector<std::string>* string_result) const override;
    std::string DebugString() const override { return response_->DebugString(); }

private:
    InferResultKyTensor(
            std::shared_ptr<inference::ModelInferResponse> response,
            Error& request_status);
    InferResultKyTensor(
            std::shared_ptr<inference::ModelStreamInferResponse> response);

    std::map<std::string, const inference::ModelInferResponse::InferOutputTensor*>
        output_name_to_tensor_map_;
    std::map<std::string, std::pair<const uint8_t*, const uint32_t>>
        output_name_to_buffer_map_;

    std::shared_ptr<inference::ModelInferResponse> response_;
    std::shared_ptr<inference::ModelStreamInferResponse> stream_response_;
    Error request_status_;
    bool is_final_response_{true};
    bool is_null_response_{false};
};

Error
InferResultKyTensor::Create(
    InferResult** infer_result,
    std::shared_ptr<inference::ModelInferResponse> response,
    Error& request_status)
{
    *infer_result = reinterpret_cast<InferResult*>(
        new InferResultKyTensor(response, request_status));
    return Error::Success;
}

Error
InferResultKyTensor::Create(
    InferResult** infer_result,
    std::shared_ptr<inference::ModelStreamInferResponse> response)
{
    *infer_result = reinterpret_cast<InferResult*>(new InferResultKyTensor(response));
    return Error::Success;
}

Error
InferResultKyTensor::RequestStatus() const
{
    return request_status_;
}

Error
InferResultKyTensor::ModelName(std::string* name) const
{
    *name = response_->model_name();
    return Error::Success;
}

Error
InferResultKyTensor::ModelVersion(std::string* version) const
{
    *version = response_->model_version();
    return Error::Success;
}

Error
InferResultKyTensor::Id(std::string* id) const
{
    *id = response_->id();
    return Error::Success;
}

Error
InferResultKyTensor::Shape(
    const std::string& output_name, std::vector<int64_t>* shape) const
{
    shape->clear();
    auto it = output_name_to_tensor_map_.find(output_name);
    if (it != output_name_to_tensor_map_.end()) {
    for (const auto dim : it->second->shape()) {
        shape->push_back(dim);
    }
    } else {
    return Error(
        "The response does not contain shape for output name '" + output_name +
        "'", KYTENSOR_ERROR_NOT_FOUND);
    }
    return Error::Success;
}

Error
InferResultKyTensor::Datatype(
    const std::string& output_name, std::string* datatype) const
{
    auto it = output_name_to_tensor_map_.find(output_name);
    if (it != output_name_to_tensor_map_.end()) {
    *datatype = it->second->datatype();
    } else {
    return Error(
        "The response does not contain datatype for output name '" +
        output_name + "'", KYTENSOR_ERROR_NOT_FOUND);
    }
    return Error::Success;
}


Error
InferResultKyTensor::RawData(
    const std::string& output_name, const uint8_t** buf,
    size_t* byte_size) const
{
    auto it = output_name_to_buffer_map_.find(output_name);
    if (it != output_name_to_buffer_map_.end()) {
    *buf = it->second.first;
    *byte_size = it->second.second;
    } else {
    return Error(
        "The response does not contain results for output name '" +
        output_name + "'", KYTENSOR_ERROR_NOT_FOUND);
    }

    return Error::Success;
}

Error
InferResultKyTensor::IsFinalResponse(bool* is_final_response) const
{
    if (is_final_response == nullptr) {
    return Error("is_final_response cannot be nullptr", KYTENSOR_ERROR_INVALID_ARG);
    }
    *is_final_response = is_final_response_;
    return Error::Success;
}

Error
InferResultKyTensor::IsNullResponse(bool* is_null_response) const
{
    if (is_null_response == nullptr) {
    return Error("is_null_response cannot be nullptr", KYTENSOR_ERROR_INVALID_ARG);
    }
    *is_null_response = is_null_response_;
    return Error::Success;
}

Error
InferResultKyTensor::StringData(
    const std::string& output_name,
    std::vector<std::string>* string_result) const
{
    std::string datatype;
    Error err = Datatype(output_name, &datatype);
    if (!err.IsOk()) {
    return err;
    }
    if (datatype.compare("BYTES") != 0) {
    return Error(
        "This function supports tensors with datatype 'BYTES', requested "
        "output tensor '" +
        output_name + "' with datatype '" + datatype + "'", KYTENSOR_ERROR_UNAVAILABLE);
    }

    const uint8_t* buf;
    size_t byte_size;
    err = RawData(output_name, &buf, &byte_size);
    string_result->clear();
    if (byte_size != 0) {
    size_t buf_offset = 0;
    while (byte_size > buf_offset) {
        const uint32_t element_size =
            *(reinterpret_cast<const uint32_t*>(buf + buf_offset));
        string_result->emplace_back(
            reinterpret_cast<const char*>(
                buf + buf_offset + sizeof(element_size)),
            element_size);
        buf_offset += (sizeof(element_size) + element_size);
    }
    } else {
    auto it = output_name_to_tensor_map_.find(output_name);
    for (const auto& element : it->second->contents().bytes_contents()) {
        string_result->push_back(element);
    }
    }

    return Error::Success;
}

InferResultKyTensor::InferResultKyTensor(
    std::shared_ptr<inference::ModelInferResponse> response,
    Error& request_status)
    : response_(response), request_status_(request_status)
{
    uint32_t index = 0;
    for (const auto& output : response_->outputs()) {
    output_name_to_tensor_map_[output.name()] = &output;
    const uint8_t* buf =
        (uint8_t*)&(response_->raw_output_contents()[index][0]);
    const uint32_t byte_size = response_->raw_output_contents()[index].size();
    output_name_to_buffer_map_.insert(
        std::make_pair(output.name(), std::make_pair(buf, byte_size)));
    index++;
    }
    const auto& is_final_response_itr{
        response_->parameters().find("triton_final_response")};
    if (is_final_response_itr != response_->parameters().end()) {
    is_final_response_ = is_final_response_itr->second.bool_param();
    }
    is_null_response_ = response_->outputs().empty() && is_final_response_;
}

InferResultKyTensor::InferResultKyTensor(
    std::shared_ptr<inference::ModelStreamInferResponse> stream_response)
    : stream_response_(stream_response)
{
    request_status_ = Error(stream_response_->error_message(),
                            KYTENSOR_ERROR_INTERNAL);
    response_.reset(
        stream_response->mutable_infer_response(),
        [](inference::ModelInferResponse*) {});
    uint32_t index = 0;
    for (const auto& output : response_->outputs()) {
    output_name_to_tensor_map_[output.name()] = &output;
    const uint8_t* buf =
        (uint8_t*)&(response_->raw_output_contents()[index][0]);
    const uint32_t byte_size = response_->raw_output_contents()[index].size();
    output_name_to_buffer_map_.insert(
        std::make_pair(output.name(), std::make_pair(buf, byte_size)));
    index++;
    }
    const auto& is_final_response_itr{
        response_->parameters().find("triton_final_response")};
    if (is_final_response_itr != response_->parameters().end()) {
    is_final_response_ = is_final_response_itr->second.bool_param();
    }
    is_null_response_ = response_->outputs().empty() && is_final_response_;
}

//==============================================================================


KyTensorImpl::KyTensorImpl(bool verbose)
    : codec_(std::bind(&ks::KytensorDispatcher::RecieveKytensorMessage, &dispatcher_, _1, _2, _3, _4)),
      dispatcher_(std::bind(&KyTensorImpl::OnUnknownMessage, this, _1, _2, _3)),
      send_time_(0)
{
    dispatcher_.RegisterMessageCallback<inference::Empty>(
        std::bind(&KyTensorImpl::OnEmpty, this, _1, _2, _3, _4));

    dispatcher_.RegisterMessageCallback<inference::ServerLiveResponse>(
        [this](const struct bufferevent* bev, const ks::MessageHeader& header,
                const ServerLiveResponsePtr& msg, int64_t receive_time) {
            HandleResponse(bev, header, msg, receive_time);
        });

    dispatcher_.RegisterMessageCallback<inference::ServerReadyResponse>(
        [this](const struct bufferevent* bev, const ks::MessageHeader& header,
                const ServerReadyResponsePtr& msg, int64_t receive_time) {
            HandleResponse(bev, header, msg, receive_time);
        });

    dispatcher_.RegisterMessageCallback<inference::ModelReadyResponse>(
        [this](const struct bufferevent* bev, const ks::MessageHeader& header,
                const ModelReadyResponsePtr& msg, int64_t receive_time) {
            HandleResponse(bev, header, msg, receive_time);
        });

    dispatcher_.RegisterMessageCallback<inference::ServerMetadataResponse>(
        [this](const struct bufferevent* bev, const ks::MessageHeader& header,
                const ServerMetadataResponsePtr& msg, int64_t receive_time) {
            HandleResponse(bev, header, msg, receive_time);
        });

    dispatcher_.RegisterMessageCallback<inference::ModelMetadataResponse>(
        [this](const struct bufferevent* bev, const ks::MessageHeader& header,
                const ModelMetadataResponsePtr& msg, int64_t receive_time) {
            HandleResponse(bev, header, msg, receive_time);
        });

    dispatcher_.RegisterMessageCallback<inference::ModelConfigResponse>(
        [this](const struct bufferevent* bev, const ks::MessageHeader& header,
                const ModelConfigResponsePtr& msg, int64_t receive_time) {
            HandleResponse(bev, header, msg, receive_time);
        });

    dispatcher_.RegisterMessageCallback<inference::RepositoryIndexResponse>(
        [this](const struct bufferevent* bev, const ks::MessageHeader& header,
                const RepositoryIndexResponsePtr& msg, int64_t receive_time) {
            HandleResponse(bev, header, msg, receive_time);
        });

    dispatcher_.RegisterMessageCallback<inference::RepositoryModelLoadResponse>(
        [this](const struct bufferevent* bev, const ks::MessageHeader& header,
                const RepositoryModelLoadResponsePtr& msg, int64_t receive_time) {
            HandleResponse(bev, header, msg, receive_time);
        });

    dispatcher_.RegisterMessageCallback<inference::RepositoryModelUnloadResponse>(
        [this](const struct bufferevent* bev, const ks::MessageHeader& header,
                const RepositoryModelUnloadResponsePtr& msg, int64_t receive_time) {
            HandleResponse(bev, header, msg, receive_time);
        });

    dispatcher_.RegisterMessageCallback<inference::ModelStatisticsResponse>(
        [this](const struct bufferevent* bev, const ks::MessageHeader& header,
                const ModelStatisticsResponsePtr& msg, int64_t receive_time) {
            HandleResponse(bev, header, msg, receive_time);
        });

    //infer
    dispatcher_.RegisterMessageCallback<inference::ModelInferResponse>(
        [this](const struct bufferevent* bev, const ks::MessageHeader& header,
                const ModelInferResponsePtr& msg, int64_t receive_time) {

            if (header.msg_infer_type == ks::MESSAGE_INFER_ASYNC) {
                Error request_status = Error::Success;
                InferResult* async_result = nullptr;
                InferResultKyTensor::Create(&async_result, msg, request_status);
                KyTensorClient::OnCompleteFn callback = nullptr;
                {
                    std::unique_lock<std::mutex> lock(mutex_async_request_);
                    auto it = infer_callbacks_.find(header.msg_id);
                    if (it != infer_callbacks_.end()) {
                        callback = it->second;
                        infer_callbacks_.erase(it);
                    }
                    async_recv_num_++;
                }

                LOGF_DBG("async infer: response id: %d, status: %d, callback num: %d\n",
                         header.msg_id, header.status, async_recv_num_);
                if (callback) {
                    callback(async_result);
                }

            } else {
                HandleResponse(bev, header, msg, receive_time);
            }
        });

    //stream
    dispatcher_.RegisterMessageCallback<inference::ModelStreamInferResponse>(
        [this](const struct bufferevent* bev, const ks::MessageHeader& header,
                const std::shared_ptr<inference::ModelStreamInferResponse>& msg, int64_t receive_time) {

                Error request_status = Error::Success;
                InferResult* stream_result = nullptr;
                stream_recv_num_++;
                LOGF_DBG("ModelStreamInferResponse: %s\n", msg->DebugString().c_str());

                InferResultKyTensor::Create(&stream_result, msg);
                stream_callback_(stream_result);
        });

    //start
    is_started_ = Start(verbose);

    stream_running_ = false;
}

KyTensorImpl::~KyTensorImpl()
{
    running_ = false;

    if (base_) {
        event_base_loopbreak(base_);
    }

    if (event_loop_thread_.joinable())  {
        event_loop_thread_.join();
    }

    if(bev_) {
        bufferevent_free(bev_);
    }
    event_base_free(base_);
    LOGF_DBG("USD client exit\n");
}

Error
KyTensorImpl::Create(std::unique_ptr<KyTensorImpl>* client, bool verbose)
{
    client->reset(new KyTensorImpl(verbose));
    return Error::Success;
}

void KyTensorImpl::OnUnknownMessage(const struct bufferevent* bev, const MessagePtr& message, int64_t receive_time) {
    (void)bev;
    (void)receive_time;
    LOG_INF("%s: Received unknown message type: %s\n", __func__, message->GetTypeName().c_str());
}

void KyTensorImpl::OnEmpty(const struct bufferevent* bev, const ks::MessageHeader& header, const EmptyPtr& msg, int64_t receive_time) {
    (void)bev;
    (void)receive_time;
    LOGF_DBG("Received empty message, id: %d\n", header.msg_id);

    callback_(msg);
}

void KyTensorImpl::SyncSendMessage(const std::string& content) {
    uint64_t send_time = t_us();
    kytensor::server::MessageHeader header;

    {
        std::unique_lock<std::mutex> lock(mutex_request_);
        header.msg_id = ++msg_id_;
        header.status = 0;
    }

    inference::ServerLiveRequest reqest;
    response_.AddResponseId(msg_id_);
    codec_.SendMessage(bev_, header, reqest);

    // Wait for response
    ks::KytensorMessagePtr res = response_.RecvResponse(msg_id_);
    response_.RemoveResponseId(msg_id_);

    uint64_t receive_us = t_us();
    uint64_t round_trip_time = receive_us - send_time;
    LOGF_DBG("%s: id: %d, rtt: %ld\n", __func__, msg_id_, round_trip_time);
}

void KyTensorImpl::AsyncSendMessage(KyTensorClient::OnCompleteFunc callback, const int id) {
    kytensor::server::MessageHeader header;

    {
        std::unique_lock<std::mutex> lock(mutex_request_);
        header.msg_id = id;
        header.status = 0;
    }

    if (callback == nullptr) {
        LOGF_ERR("callback function must be provide along with AsyncSendMessage() call.\n");
        return;
    }

    callback_ = std::move(callback);

    inference::Empty empty;
    empty.set_id(id);

    LOG_DBG("%s: send empty message %d\n", __func__, id);
    codec_.SendMessage(bev_, header, empty);
}

void KyTensorImpl::EventLoop() {

    base_ = event_base_new();
    if (!base_) {
        LOG_ERR("%s: Could not initialize libevent!\n", __func__);
        // notify failure if someone is waiting
        if (start_promise_) {
            try { start_promise_->set_value(false); } catch(...) {}
            start_promise_.reset();
        }
        return;
    }

    // Create the Unix domain socket address
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);

    bev_ = bufferevent_socket_new(base_, -1, BEV_OPT_CLOSE_ON_FREE  | BEV_OPT_THREADSAFE);
    if (!bev_) {
        LOG_ERR("%s: bufferevent_socket_new failed\n", __func__);
        bev_ = nullptr;
        // notify failure
        if (start_promise_) {
            try { start_promise_->set_value(false); } catch(...) {}
            start_promise_.reset();
        }
        return;
    }

    bufferevent_setcb(bev_,
                      ReceiveMessageCallback,
                      OnWrite,
                      OnConnect,
                      this);
    //set read/write watermark
    bufferevent_setwatermark(bev_, EV_READ, 18, 0);
    bufferevent_setwatermark(bev_, EV_WRITE, 18, 0);
    bufferevent_enable(bev_, EV_READ|EV_WRITE);

    // Connect to the server
    if (bufferevent_socket_connect(bev_, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        LOGF_ERR("connect failed\n");
        bufferevent_free(bev_);
        bev_ = nullptr;
        // notify failure
        if (start_promise_) {
            try { start_promise_->set_value(false); } catch(...) {}
            start_promise_.reset();
        }
        return;
    }

    // notify Start() that EventLoop created bev_ (successful init)
    if (start_promise_) {
        auto sp = start_promise_;
        // avoid double set by resetting member before set_value
        start_promise_.reset();
        try {
            sp->set_value(true);
        } catch(...) {
            // ignore
        }
    }

    LOG_INF("%s: Connecting to server...\n", __func__);

    event_base_dispatch(base_);
}

void KyTensorImpl::EventLoopTcp()
{
    base_ = event_base_new();
    if (!base_) {
        LOG_ERR("%s: Could not initialize libevent!\n", __func__);
        return;
    }

    // Create TCP socket address (default localhost:5555)
    const char* SERVER_HOST = "127.0.0.1";
    const uint16_t SERVER_PORT = 5555;
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(SERVER_PORT);
    if (evutil_inet_pton(AF_INET, SERVER_HOST, &addr.sin_addr) != 1) {
        LOG_ERR("%s: inet_pton failed for %s\n", __func__, SERVER_HOST);
        event_base_free(base_);
        return;
    }

    bev_ = bufferevent_socket_new(base_, -1, BEV_OPT_CLOSE_ON_FREE  | BEV_OPT_THREADSAFE);
    if (!bev_) {
        LOG_ERR("%s: bufferevent_socket_new failed\n", __func__);
        event_base_free(base_);
        return;
    }

    bufferevent_setcb(bev_,
                      ReceiveMessageCallback,
                      OnWrite,
                      OnConnect,
                      this);
    bufferevent_setwatermark(bev_, EV_READ, 18, 0);
    bufferevent_setwatermark(bev_, EV_WRITE, 18, 0);
    bufferevent_enable(bev_, EV_READ|EV_WRITE);

    // Connect to TCP server
    if (bufferevent_socket_connect(bev_, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        LOG_ERR("%s: connect failed to %s:%u\n", __func__, SERVER_HOST, SERVER_PORT);
        bufferevent_free(bev_);
        event_base_free(base_);
        return;
    }

    LOG_INF("%s: Connecting to server %s:%u...\n", __func__, SERVER_HOST, SERVER_PORT);
    event_base_dispatch(base_);
}

Error
KyTensorImpl::PreRunProcessing(
    const InferOptions& options, const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs)
{
    // Populate the request protobuf
    infer_request_.set_model_name(options.model_name_);
    infer_request_.set_model_version(options.model_version_);
    infer_request_.set_id(options.request_id_);

    infer_request_.mutable_parameters()->clear();
    (*infer_request_.mutable_parameters())["triton_enable_empty_final_response"]
        .set_bool_param(options.triton_enable_empty_final_response_);
    if ((options.sequence_id_ != 0) || (options.sequence_id_str_ != "")) {
        if (options.sequence_id_ != 0) {
        (*infer_request_.mutable_parameters())["sequence_id"].set_int64_param(
            options.sequence_id_);
        } else {
        (*infer_request_.mutable_parameters())["sequence_id"].set_string_param(
            options.sequence_id_str_);
        }
        (*infer_request_.mutable_parameters())["sequence_start"].set_bool_param(
            options.sequence_start_);
        (*infer_request_.mutable_parameters())["sequence_end"].set_bool_param(
            options.sequence_end_);
    }
    if (options.priority_ != 0) {
        (*infer_request_.mutable_parameters())["priority"].set_uint64_param(
            options.priority_);
    }

    if (options.server_timeout_ != 0) {
        (*infer_request_.mutable_parameters())["timeout"].set_int64_param(
            options.server_timeout_);
    }


    for (auto& param : options.request_parameters) {
        if (param.second.type == "string") {
        (*infer_request_.mutable_parameters())[param.first].set_string_param(
            param.second.value);
        } else if (param.second.type == "int") {
        (*infer_request_.mutable_parameters())[param.first].set_int64_param(
            std::stoi(param.second.value));
        } else if (param.second.type == "bool") {
        bool val = false;
        if (param.second.value == "true") {
            val = true;
        }
        (*infer_request_.mutable_parameters())[param.first].set_bool_param(val);
        }
    }

    int index = 0;
    infer_request_.mutable_raw_input_contents()->Clear();
    for (const auto input : inputs) {
        // Add new InferInputTensor submessages only if required, otherwise
        // reuse the submessages already available.
        auto grpc_input = (infer_request_.inputs().size() <= index)
                            ? infer_request_.add_inputs()
                            : infer_request_.mutable_inputs()->Mutable(index);

        if (input->IsSharedMemory()) {
        // The input contents must be cleared when using shared memory.
        grpc_input->Clear();
        }

        grpc_input->set_name(input->Name());
        grpc_input->mutable_shape()->Clear();
        for (const auto dim : input->Shape()) {
        grpc_input->mutable_shape()->Add(dim);
        }
        grpc_input->set_datatype(input->Datatype());

        input->PrepareForRequest();
        grpc_input->mutable_parameters()->clear();
        if (input->IsSharedMemory()) {
        std::string region_name;
        size_t offset;
        size_t byte_size;
        input->SharedMemoryInfo(&region_name, &byte_size, &offset);

        (*grpc_input->mutable_parameters())["shared_memory_region"]
            .set_string_param(region_name);
        (*grpc_input->mutable_parameters())["shared_memory_byte_size"]
            .set_int64_param(byte_size);
        if (offset != 0) {
            (*grpc_input->mutable_parameters())["shared_memory_offset"]
                .set_int64_param(offset);
        }
        } else {
        bool end_of_input = false;
        std::string* raw_contents = infer_request_.add_raw_input_contents();
        size_t content_size;
        input->ByteSize(&content_size);
        raw_contents->reserve(content_size);
        raw_contents->clear();
        while (!end_of_input) {
            const uint8_t* buf;
            size_t buf_size;
            input->GetNext(&buf, &buf_size, &end_of_input);
            if (buf != nullptr) {
            raw_contents->append(reinterpret_cast<const char*>(buf), buf_size);
            }
        }
        }
        index++;
    }

    // Remove extra InferInputTensor submessages, that are not required for
    // this request.
    while (index < infer_request_.inputs().size()) {
        infer_request_.mutable_inputs()->RemoveLast();
    }

    index = 0;
    for (const auto routput : outputs) {
        // Add new InferRequestedOutputTensor submessage only if required, otherwise
        // reuse the submessages already available.
        auto grpc_output = (infer_request_.outputs().size() <= index)
                            ? infer_request_.add_outputs()
                            : infer_request_.mutable_outputs()->Mutable(index);
        grpc_output->Clear();
        grpc_output->set_name(routput->Name());
        size_t class_count = routput->ClassificationCount();
        if (class_count != 0) {
        (*grpc_output->mutable_parameters())["classification"].set_int64_param(
            class_count);
        }
        if (routput->IsSharedMemory()) {
        std::string region_name;
        size_t offset;
        size_t byte_size;
        routput->SharedMemoryInfo(&region_name, &byte_size, &offset);
        (*grpc_output->mutable_parameters())["shared_memory_region"]
            .set_string_param(region_name);
        (*grpc_output->mutable_parameters())["shared_memory_byte_size"]
            .set_int64_param(byte_size);
        if (offset != 0) {
            (*grpc_output->mutable_parameters())["shared_memory_offset"]
                .set_int64_param(offset);
        }
        }
        index++;
    }

    // Remove extra InferRequestedOutputTensor submessages, that are not required
    // for this request.
    while (index < infer_request_.outputs().size()) {
        infer_request_.mutable_outputs()->RemoveLast();
    }

    if (infer_request_.ByteSizeLong() > INT_MAX) {
        size_t request_size = infer_request_.ByteSizeLong();
        infer_request_.Clear();
        return Error(
            "Request has byte size " + std::to_string(request_size) +
            " which exceed Kytensor byte size limit " + std::to_string(INT_MAX) +
            ".", KYTENSOR_ERROR_UNAVAILABLE);
    }

    return Error::Success;
}

bool KyTensorImpl::Start(bool verbose)
{
    if (verbose) {
        common_log_set_verbosity_thold(2);
    }
    common_log_set_prefix(common_log_main(), true);
    common_log_set_timestamps(common_log_main(), true);
    evthread_use_pthreads();

    if (!event_loop_thread_.joinable()) {
         // create promise/future for sync start
         start_promise_ = std::make_shared<std::promise<bool>>();
         std::future<bool> fut = start_promise_->get_future();

        event_loop_thread_ = std::thread(&KyTensorImpl::EventLoop, this);

        // wait for EventLoop to notify (timeout optional)
        auto status = fut.wait_for(std::chrono::seconds(5));
        if (status != std::future_status::ready) {
            LOGF_ERR("timeout waiting for EventLoop initialization\n");
            return false;
        }
        bool notified = false;
        try {
            notified = fut.get();
        } catch (...) {
            LOGF_ERR("Start: exception while getting start future\n");
            return false;
        }

        if (!notified) {
            LOGF_ERR("EventLoop reported failure during init\n");
            return false;
        }

        if (bev_ == nullptr) {
            LOGF_ERR("kytenser server is not live\n");
            return false;
        }

        //bad way to check if server is live, but we need to caculate the live check time for better timeout setting in future calls
        bool live = false;
        Error err ;
        err = IsServerLive(&live);
        if (!err.IsOk()) {
            LOGF_ERR("Server live check failed: %s\n", err.Message().c_str());
            return false;
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 20; i++) {
            err = IsServerLive(&live, 100);
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        int  avg_time = duration / 20;
        if (avg_time > 300 || avg_time <= 0)
            avg_time = 300;

        kmax_delay_us_ = avg_time;
        LOGF_INF("Server live check time: %d us\n", avg_time);

        return true;
    }

    return false;
}

void KyTensorImpl::Stop()
{
    if (base_) {
        event_base_loopbreak(base_);
    }

    if (event_loop_thread_.joinable())  {
        event_loop_thread_.join();
    }
    LOG_INF("%s: USD client stopped\n", __func__);
}

void KyTensorImpl::ReceiveMessageCallback(bufferevent *bev, void *arg)
{
    KyTensorImpl* client = static_cast<KyTensorImpl*>(arg);
    client->codec_.ReceiveMessage(bev, arg);
}

void KyTensorImpl::OnWrite(bufferevent *bev, void *arg)
{
    (void)bev;
    (void)arg;
}

void KyTensorImpl::OnConnect(bufferevent *bev, short events, void *arg)
{
    (void)bev;
    KyTensorImpl* client = static_cast<KyTensorImpl*>(arg);
    if (events & BEV_EVENT_CONNECTED) {
        LOG_INF("%s: Connected to server\n", __func__);
    } else if (events & BEV_EVENT_ERROR) {
        LOG_ERR("%s: can not connect to server\n", __func__);
        event_base_loopexit(client->base_, NULL);
    } else if (events & BEV_EVENT_EOF) {
        LOG_INF("%s: Server disconnected\n", __func__);
        event_base_loopexit(client->base_, NULL);
    }
}

Error
KyTensorImpl::IsServerLive(bool* live, const uint64_t timeout_ms)
{
    Error err;

    if (live == nullptr) {
        return Error("live pointer is null", KYTENSOR_ERROR_INVALID_ARG);
    }

    kytensor::server::MessageHeader header;
    {
        std::unique_lock<std::mutex> lock(mutex_request_);
        header.msg_id = ++msg_id_;
        header.status = 0;
    }
    inference::ServerLiveRequest request;
    response_.AddResponseId(msg_id_);
    codec_.SendMessage(bev_, header, request);

    ks::KytensorMessagePtr res_ptr = nullptr;
    if (timeout_ms) {
        res_ptr = response_.RecvResponse(msg_id_, timeout_ms);
    } else {
        res_ptr = response_.RecvResponse(msg_id_);
    }
    response_.RemoveResponseId(msg_id_);
    if (res_ptr == nullptr) {
        err = Error("No response received from server within the specified timeout.", KYTENSOR_ERROR_UNAVAILABLE);
        return err;
    }

    LOGF_DBG("response id: %d, status: %d\n", res_ptr->msg_id, res_ptr->status);
    err.SetStatus(res_ptr->status);

    auto res = std::dynamic_pointer_cast<inference::ServerLiveResponse>(res_ptr->message);
    if (res != nullptr) {
        *live = res->live();
    } else {
        err = Error("message is nulptr", KYTENSOR_ERROR_UNAVAILABLE);
    }

    return err;
}

Error
KyTensorImpl::IsServerReady(bool* ready, const uint64_t timeout_ms)
{
    Error err;
    kytensor::server::MessageHeader header;
    {
        std::unique_lock<std::mutex> lock(mutex_request_);
        header.msg_id = ++msg_id_;
        header.status = 0;
    }

    inference::ServerReadyRequest request;
    response_.AddResponseId(msg_id_);
    codec_.SendMessage(bev_, header, request);

    ks::KytensorMessagePtr res_ptr = nullptr;
    if (timeout_ms) {
        res_ptr = response_.RecvResponse(msg_id_, timeout_ms);
    } else {
        res_ptr = response_.RecvResponse(msg_id_);
    }
    response_.RemoveResponseId(msg_id_);
    if (res_ptr == nullptr) {
        err = Error("No response received from server within the specified timeout.", KYTENSOR_ERROR_UNAVAILABLE);
        return err;
    }

    LOGF_DBG("response id: %d, status: %d\n", res_ptr->msg_id, res_ptr->status);
    err.SetStatus(res_ptr->status);

    auto res = std::dynamic_pointer_cast<inference::ServerReadyResponse>(res_ptr->message);
    if (res != nullptr) {
        *ready = res->ready();
        LOGF_DBG("Server Ready: %d\n", *ready);
    } else {
        err = Error("message is nulptr", KYTENSOR_ERROR_UNAVAILABLE);
    }

    return err;
}

Error
KyTensorImpl::IsModelReady(
    bool* ready, const std::string& model_name,
    const std::string& model_version, const uint64_t timeout_ms)
{
    Error err;
    kytensor::server::MessageHeader header;
    {
        std::unique_lock<std::mutex> lock(mutex_request_);
        header.msg_id = ++msg_id_;
        header.status = 0;
    }

    inference::ModelReadyRequest request;
    request.set_name(model_name);
    request.set_version(model_version);
    response_.AddResponseId(msg_id_);
    codec_.SendMessage(bev_, header, request);

    ks::KytensorMessagePtr res_ptr = nullptr;
    if (timeout_ms) {
        res_ptr = response_.RecvResponse(msg_id_, timeout_ms);
    } else {
        res_ptr = response_.RecvResponse(msg_id_);
    }
    response_.RemoveResponseId(msg_id_);
    if (res_ptr == nullptr) {
        err = Error("No response received from server within the specified timeout.", KYTENSOR_ERROR_UNAVAILABLE);
        return err;
    }

    LOGF_DBG("response id: %d, status: %d\n", res_ptr->msg_id, res_ptr->status);
    err.SetStatus(res_ptr->status);

    auto res = std::dynamic_pointer_cast<inference::ModelReadyResponse>(res_ptr->message);
    if (res != nullptr) {
        *ready = res->ready();
        LOGF_DBG("Model Ready : name: %s, ready: %d\n", model_name.c_str(), *ready);
    } else {
        err = Error("message is nulptr", KYTENSOR_ERROR_UNAVAILABLE);
    }

    return err;
}

Error
KyTensorImpl::ServerMetadata(
    inference::ServerMetadataResponse* server_metadata,
    const uint64_t timeout_ms)
{
    server_metadata->Clear();
    Error err;
    kytensor::server::MessageHeader header;
    {
        std::unique_lock<std::mutex> lock(mutex_request_);
        header.msg_id = ++msg_id_;
        header.status = 0;
    }

    inference::ServerMetadataRequest request;
    response_.AddResponseId(msg_id_);
    codec_.SendMessage(bev_, header, request);

    ks::KytensorMessagePtr res_ptr = nullptr;
    if (timeout_ms) {
        res_ptr = response_.RecvResponse(msg_id_, timeout_ms);
    } else {
        res_ptr = response_.RecvResponse(msg_id_);
    }
    response_.RemoveResponseId(msg_id_);
    if (res_ptr == nullptr) {
        err = Error("No response received from server within the specified timeout.", KYTENSOR_ERROR_UNAVAILABLE);
        return err;
    }

    LOGF_DBG("response id: %d, status: %d\n", res_ptr->msg_id, res_ptr->status);
    err.SetStatus(res_ptr->status);

    auto res = std::dynamic_pointer_cast<inference::ServerMetadataResponse>(res_ptr->message);
    if (res != nullptr) {
        server_metadata->CopyFrom(*res);
        LOGF_DBG("Server metadata: %s\n", server_metadata->DebugString().c_str());
    } else {
        err = Error("message is nulptr", KYTENSOR_ERROR_UNAVAILABLE);
    }

    return err;
}

Error
KyTensorImpl::ModelMetadata(
    inference::ModelMetadataResponse* model_metadata,
    const std::string& model_name, const std::string& model_version,
    const uint64_t timeout_ms)
{
    model_metadata->Clear();
    Error err;
    kytensor::server::MessageHeader header;
    {
        std::unique_lock<std::mutex> lock(mutex_request_);
        header.msg_id = ++msg_id_;
        header.status = 0;
    }

    inference::ModelMetadataRequest request;
    request.set_name(model_name);
    request.set_version(model_version);
    response_.AddResponseId(msg_id_);
    codec_.SendMessage(bev_, header, request);

    ks::KytensorMessagePtr res_ptr = nullptr;
    if (timeout_ms) {
        res_ptr = response_.RecvResponse(msg_id_, timeout_ms);
    } else {
        res_ptr = response_.RecvResponse(msg_id_);
    }
    response_.RemoveResponseId(msg_id_);
    if (res_ptr == nullptr) {
        err = Error("No response received from server within the specified timeout.", KYTENSOR_ERROR_UNAVAILABLE);
        return err;
    }

    LOGF_DBG("response id: %d, status: %d\n", res_ptr->msg_id, res_ptr->status);
    err.SetStatus(res_ptr->status);

    auto res = std::dynamic_pointer_cast<inference::ModelMetadataResponse>(res_ptr->message);
    if (res != nullptr) {
        model_metadata->CopyFrom(*res);
        LOGF_DBG("Model metadata: %s\n", model_metadata->DebugString().c_str());
    } else {
        err = Error("message is nulptr", KYTENSOR_ERROR_UNAVAILABLE);
    }

    return err;
}

Error
KyTensorImpl::ModelConfig(
    inference::ModelConfigResponse* model_config, const std::string& model_name,
    const std::string& model_version, const uint64_t timeout_ms)
{
    if (model_config == nullptr) {
        return Error("model_config pointer is null", KYTENSOR_ERROR_INVALID_ARG);
    }

    model_config->Clear();
    Error err;
    kytensor::server::MessageHeader header;
    {
        std::unique_lock<std::mutex> lock(mutex_request_);
        header.msg_id = ++msg_id_;
        header.status = 0;
    }

    inference::ModelConfigRequest request;
    request.set_name(model_name);
    request.set_version(model_version);
    response_.AddResponseId(msg_id_);
    codec_.SendMessage(bev_, header, request);

    ks::KytensorMessagePtr res_ptr = nullptr;
    if (timeout_ms) {
        res_ptr = response_.RecvResponse(msg_id_, timeout_ms);
    } else {
        res_ptr = response_.RecvResponse(msg_id_);
    }
    response_.RemoveResponseId(msg_id_);
    if (res_ptr == nullptr) {
        err = Error("No response received from server within the specified timeout.", KYTENSOR_ERROR_UNAVAILABLE);
        return err;
    }

    LOGF_DBG("response id: %d, status: %d\n", res_ptr->msg_id, res_ptr->status);
    err.SetStatus(res_ptr->status);

    auto res = std::dynamic_pointer_cast<inference::ModelConfigResponse>(res_ptr->message);
    if (res != nullptr) {
        model_config->CopyFrom(*res);
        LOGF_DBG("Model metadata: %s\n", model_config->DebugString().c_str());
    } else {
        err = Error("message is nulptr", KYTENSOR_ERROR_UNAVAILABLE);
    }

    return err;
}

Error
KyTensorImpl::ModelRepositoryIndex(
    inference::RepositoryIndexResponse* repository_index,
    const uint64_t timeout_ms)
{
    repository_index->Clear();
    Error err;
    kytensor::server::MessageHeader header;
    {
        std::unique_lock<std::mutex> lock(mutex_request_);
        header.msg_id = ++msg_id_;
        header.status = 0;
    }

    inference::RepositoryIndexRequest request;
    response_.AddResponseId(msg_id_);
    codec_.SendMessage(bev_, header, request);

    ks::KytensorMessagePtr res_ptr = nullptr;
    if (timeout_ms) {
        res_ptr = response_.RecvResponse(msg_id_, timeout_ms);
    } else {
        res_ptr = response_.RecvResponse(msg_id_);
    }
    response_.RemoveResponseId(msg_id_);
    if (res_ptr == nullptr) {
        err = Error("No response received from server within the specified timeout.", KYTENSOR_ERROR_UNAVAILABLE);
        return err;
    }

    LOGF_DBG("response id: %d, status: %d\n", res_ptr->msg_id, res_ptr->status);
    err.SetStatus(res_ptr->status);

    auto res = std::dynamic_pointer_cast<inference::RepositoryIndexResponse>(res_ptr->message);
    if (res != nullptr) {
        repository_index->CopyFrom(*res);
        LOGF_DBG("Model metadata: %s\n", repository_index->DebugString().c_str());
    } else {
        err = Error("message is nulptr", KYTENSOR_ERROR_UNAVAILABLE);
    }

    return err;
}

Error
KyTensorImpl::LoadModel(
    const std::string& model_name, const std::string& config,
    const std::map<std::string, std::vector<char>>& files,
    const uint64_t timeout_ms)
{
    Error err;
    kytensor::server::MessageHeader header;
    {
        std::unique_lock<std::mutex> lock(mutex_request_);
        header.msg_id = ++msg_id_;
        header.status = 0;
    }
    LOGF_DBG("load model, timeout: %ld\n", timeout_ms);
    inference::RepositoryModelLoadRequest request;
    request.set_model_name(model_name);
    if (!config.empty()) {
        (*request.mutable_parameters())["config"].set_string_param(config);
    }
    for (const auto& file : files) {
        (*request.mutable_parameters())[file.first].set_bytes_param(
            file.second.data(), file.second.size());
    }
    response_.AddResponseId(msg_id_);
    codec_.SendMessage(bev_, header, request);

    ks::KytensorMessagePtr res_ptr = nullptr;
    if (timeout_ms) {
        res_ptr = response_.RecvResponse(msg_id_, timeout_ms);
    } else {
        res_ptr = response_.RecvResponse(msg_id_);
    }
    response_.RemoveResponseId(msg_id_);
    if (res_ptr == nullptr) {
        err = Error("No response received from server within the specified timeout.", KYTENSOR_ERROR_UNAVAILABLE);
        return err;
    }

    LOGF_DBG("response id: %d, status: %d\n", res_ptr->msg_id, res_ptr->status);
    err.SetStatus(res_ptr->status);

    auto res = std::dynamic_pointer_cast<inference::RepositoryModelLoadResponse>(res_ptr->message);
    if (res != nullptr) {
        LOGF_DBG("Loaded model: %s\n", model_name.c_str());
    } else {
        err = Error("message is nulptr", KYTENSOR_ERROR_UNAVAILABLE);
    }

    return err;
}

Error
KyTensorImpl::UnloadModel(const std::string& model_name, const uint64_t timeout_ms)
{
    Error err;
    kytensor::server::MessageHeader header;
    {
        std::unique_lock<std::mutex> lock(mutex_request_);
        header.msg_id = ++msg_id_;
        header.status = 0;
    }

    inference::RepositoryModelUnloadRequest request;
    request.set_model_name(model_name);
    response_.AddResponseId(msg_id_);
    codec_.SendMessage(bev_, header, request);

    ks::KytensorMessagePtr res_ptr = nullptr;
    if (timeout_ms) {
        res_ptr = response_.RecvResponse(msg_id_, timeout_ms);
    } else {
        res_ptr = response_.RecvResponse(msg_id_);
    }
    response_.RemoveResponseId(msg_id_);
    if (res_ptr == nullptr) {
        err = Error("No response received from server within the specified timeout.", KYTENSOR_ERROR_UNAVAILABLE);
        return err;
    }

    LOGF_DBG("response id: %d, status: %d\n", res_ptr->msg_id, res_ptr->status);
    err.SetStatus(res_ptr->status);

    auto res = std::dynamic_pointer_cast<inference::RepositoryModelUnloadResponse>(res_ptr->message);
    if (res != nullptr) {
        LOGF_DBG("Unloaded model: %s\n", model_name.c_str());
    } else {
        err = Error("message is nulptr", KYTENSOR_ERROR_UNAVAILABLE);
    }

    return err;
}

Error
KyTensorImpl::ModelInferenceStatistics(
    inference::ModelStatisticsResponse* infer_stat,
    const std::string& model_name, const std::string& model_version,
    const uint64_t timeout_ms)
{
    infer_stat->Clear();
    Error err;
    kytensor::server::MessageHeader header;
    {
        std::unique_lock<std::mutex> lock(mutex_request_);
        header.msg_id = ++msg_id_;
        header.status = 0;
    }

    inference::ModelStatisticsRequest request;
    request.set_name(model_name);
    request.set_version(model_version);
    response_.AddResponseId(msg_id_);
    codec_.SendMessage(bev_, header, request);

    ks::KytensorMessagePtr res_ptr = nullptr;
    if (timeout_ms) {
        res_ptr = response_.RecvResponse(msg_id_, timeout_ms);
    } else {
        res_ptr = response_.RecvResponse(msg_id_);
    }
    response_.RemoveResponseId(msg_id_);
    if (res_ptr == nullptr) {
        err = Error("No response received from server within the specified timeout.", KYTENSOR_ERROR_UNAVAILABLE);
        return err;
    }

    LOGF_DBG("response id: %d, status: %d\n", res_ptr->msg_id, res_ptr->status);
    err.SetStatus(res_ptr->status);

    auto res = std::dynamic_pointer_cast<inference::ModelStatisticsResponse>(res_ptr->message);
    if (res != nullptr) {
        infer_stat->CopyFrom(*res);
        LOGF_DBG("Infer statistics: %s\n", infer_stat->DebugString().c_str());
    } else {
        err = Error("message is nulptr", KYTENSOR_ERROR_UNAVAILABLE);
    }

    return err;
}

Error
KyTensorImpl::Infer(
    InferResult** result, const InferOptions& options,
    const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs,
    KYTENSOR_COMPRESSION_TYPE compression_type)
{
    Error err;
    kytensor::server::MessageHeader header;
    {
        std::unique_lock<std::mutex> lock(mutex_request_);
        header.msg_id = ++msg_id_;
        header.msg_type = ks::MESSAGE_TYPE_REQUEST;
        header.msg_infer_type = ks::MESSAGE_INFER_SYNC;
        header.status = 0;
    }

    err = PreRunProcessing(options, inputs, outputs);
    if (!err.IsOk()) {
        return err;
    }
    response_.AddResponseId(msg_id_);
    codec_.SendMessage(bev_, header, infer_request_);

    ks::KytensorMessagePtr res_ptr = nullptr;
    res_ptr = response_.RecvResponse(msg_id_);
    response_.RemoveResponseId(msg_id_);
    if (res_ptr == nullptr) {
        err = Error("No response received from server within the specified timeout.", KYTENSOR_ERROR_UNAVAILABLE);
        return err;
    }

    LOGF_DBG("response id: %d, status: %d\n", res_ptr->msg_id, res_ptr->status);
    err.SetStatus(res_ptr->status);

    auto res = std::dynamic_pointer_cast<inference::ModelInferResponse>(res_ptr->message);
    if (res != nullptr) {
        InferResultKyTensor::Create(result, res, err);
        LOGF_DBG("Model infer response: %s\n", res->DebugString().c_str());
    } else {
        err = Error("message is nulptr", KYTENSOR_ERROR_UNAVAILABLE);
    }

    return err;
}

Error
KyTensorImpl::AsyncInfer(
    KyTensorClient::OnCompleteFn callback, const InferOptions& options,
    const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs,
    KYTENSOR_COMPRESSION_TYPE compression_type)
{
    Error err;
    kytensor::server::MessageHeader header;

    int  distance;
    {
        std::unique_lock<std::mutex> lock(mutex_async_request_);
        header.msg_id = ++async_msg_id_;
        header.msg_type = ks::MESSAGE_TYPE_REQUEST;
        header.msg_infer_type = ks::MESSAGE_INFER_ASYNC;
        header.status = 0;
        //need to improve
        infer_callbacks_.insert({async_msg_id_, callback});

        err = PreRunProcessing(options, inputs, outputs);
        if (!err.IsOk()) {
            return err;
        }

    }

    distance = async_msg_id_ - async_recv_num_;
    static int sync_last_ema =  0;

    double ema = static_cast<double>(sync_last_ema) * (1.0 - kalpha_)
                    + static_cast<double>(distance) * kalpha_;
    sync_last_ema = static_cast<int>(std::round(ema));

    // 根据目标差值按比例计算延迟（微秒），并约束到 [0, kMaxDelayUs]
    int delta = static_cast<int>(std::round(ema)) - ktarget_distance_;
    int delay_us = 0;
    if (delta > 0) {
        delay_us = std::min(kmax_delay_us_, delta * kdelay_unit_us_);
    } else {
        delay_us = 0;
    }

    if (distance > async_last_distance_) {
        if (ema > 400)
            delay_us = 0;
        std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
    }

    async_last_distance_ = distance;


    codec_.SendMessage(bev_, header, infer_request_);

    #if 0
    static std::chrono::steady_clock::time_point last_send_time =
        std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        now - last_send_time).count();

    last_send_time = std::chrono::steady_clock::now();
    LOGF_INF("async distance: %d, ema: %.2f, delay: %d, elapsed: %ld\n", distance, ema, delay_us, elapsed);
    #endif

    return Error::Success;
}

Error
KyTensorImpl::StartStream(
    KyTensorClient::OnCompleteFn callback, bool enable_stats, uint32_t stream_timeout,
    KYTENSOR_COMPRESSION_TYPE compression_type)
{
    if (stream_running_) {
        return Error("Stream is already running", KYTENSOR_ERROR_ALREADY_EXISTS);
    }

    if (callback == nullptr) {
        return Error("callback function must be provided along wit StartStream() call", KYTENSOR_ERROR_INVALID_ARG);
    }

    stream_running_ = true;
    stream_callback_ = std::move(callback);

    LOGF_DBG("Started stream...\n");

    return Error::Success;
}

Error
KyTensorImpl::StopStream()
{
    stream_running_ = false;
    LOGF_DBG("Stopped stream...\n");
    return Error();
}

Error
KyTensorImpl::AsyncStreamInfer(
    const InferOptions& options, const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs)
{
    Error err;
    kytensor::server::MessageHeader header;

    int  distance;
    {
        std::unique_lock<std::mutex> lock(mutex_stream_request_);
        header.msg_id = ++stream_msg_id_;
        header.msg_type = ks::MESSAGE_TYPE_REQUEST;
        header.msg_infer_type = ks::MESSAGE_INFER_STREAM;
        header.status = 0;

        err = PreRunProcessing(options, inputs, outputs);
        if (!err.IsOk()) {
            return err;
        }

    }
    distance = stream_msg_id_ - stream_recv_num_;

    static int stream_last_ema =  0;

    double ema = static_cast<double>(stream_last_ema) * (1.0 - kalpha_)
                    + static_cast<double>(distance) * kalpha_;
    stream_last_ema = static_cast<int>(std::round(ema));

    // 根据目标差值按比例计算延迟（微秒），并约束到 [0, kMaxDelayUs]
    int delta = static_cast<int>(std::round(ema)) - ktarget_distance_;
    int delay_us = 0;
    if (delta > 0) {
        delay_us = std::min(kmax_delay_us_, delta * kdelay_unit_us_);
    } else {
        delay_us = 0;
    }

    if (distance > stream_last_distance_) {
        if (ema > 400)
            delay_us = 0;
        std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
    }

    stream_last_distance_ = distance;

    codec_.SendMessage(bev_, header, infer_request_);

    #if 0
    static std::chrono::steady_clock::time_point last_send_time =
        std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        now - last_send_time).count();

    last_send_time = std::chrono::steady_clock::now();
    LOGF_INF("stream distance: %d, ema: %.2f, delay: %d, elapsed: %ld\n", distance, ema, delay_us, elapsed);
    #endif

    return Error::Success;
}

//==============================================================================
// KyTensorClient wrapper implementation
//==============================================================================

KyTensorClient::~KyTensorClient() = default;

Error KyTensorClient::Create(std::unique_ptr<KyTensorClient>* client, bool verbose) {
    std::unique_ptr<KyTensorImpl> impl;
    Error err = KyTensorImpl::Create(&impl, verbose);
    if ((impl->is_started_ == false)) {
        err = Error("failed to create KytensorClient", KYTENSOR_ERROR_UNAVAILABLE);
    }

    *client = std::unique_ptr<KyTensorClient>(new KyTensorClient(std::move(impl)));

    return err;
}

KyTensorClient::KyTensorClient(std::unique_ptr<KyTensorImpl> impl)
    : pImpl(std::move(impl)) {}

bool KyTensorClient::Start(bool verbose) {
    return pImpl->Start(verbose);
}

void KyTensorClient::Stop() {
    pImpl->Stop();
}

void KyTensorClient::SyncSendMessage(const std::string& content) {
    pImpl->SyncSendMessage(content);
}

void KyTensorClient::AsyncSendMessage(OnCompleteFunc callback, const int id) {
    pImpl->AsyncSendMessage(callback, id);
}

Error KyTensorClient::IsServerLive(bool* live, const uint64_t timeout_ms) {
    return pImpl->IsServerLive(live, timeout_ms);
}

Error KyTensorClient::IsServerReady(bool* ready, const uint64_t timeout_ms) {
    return pImpl->IsServerReady(ready, timeout_ms);
}

Error KyTensorClient::IsModelReady(bool* ready, const std::string& model_name,
                                const std::string& model_version, const uint64_t timeout_ms) {
    return pImpl->IsModelReady(ready, model_name, model_version, timeout_ms);
}

Error KyTensorClient::ServerMetadata(inference::ServerMetadataResponse* server_metadata,
                                 const uint64_t timeout_ms) {
    return pImpl->ServerMetadata(server_metadata, timeout_ms);
}

Error KyTensorClient::ModelMetadata(inference::ModelMetadataResponse* model_metadata,
                                const std::string& model_name, const std::string& model_version,
                                const uint64_t timeout_ms) {
    return pImpl->ModelMetadata(model_metadata, model_name, model_version, timeout_ms);
}

Error KyTensorClient::ModelConfig(inference::ModelConfigResponse* model_config,
                                const std::string& model_name, const std::string& model_version,
                                const uint64_t timeout_ms) {
    return pImpl->ModelConfig(model_config, model_name, model_version, timeout_ms);
}

Error KyTensorClient::ModelRepositoryIndex(inference::RepositoryIndexResponse* repository_index,
                                       const uint64_t timeout_ms) {
    return pImpl->ModelRepositoryIndex(repository_index, timeout_ms);
}

Error KyTensorClient::LoadModel(const std::string& model_name,
                              const std::string& config,
                              const std::map<std::string, std::vector<char>>& files,
                              const uint64_t timeout_ms) {
    return pImpl->LoadModel(model_name, config, files, timeout_ms);
}

Error KyTensorClient::UnloadModel(const std::string& model_name, const uint64_t timeout_ms) {
    return pImpl->UnloadModel(model_name, timeout_ms);
}

Error KyTensorClient::ModelInferenceStatistics(inference::ModelStatisticsResponse* infer_stat,
                                            const std::string& model_name, const std::string& model_version,
                                            const uint64_t timeout_ms) {
    return pImpl->ModelInferenceStatistics(infer_stat, model_name, model_version, timeout_ms);
}

Error KyTensorClient::Infer(InferResult** result, const InferOptions& options,
                           const std::vector<InferInput*>& inputs,
                           const std::vector<const InferRequestedOutput*>& outputs,
                           KYTENSOR_COMPRESSION_TYPE compression_type) {
    return pImpl->Infer(result, options, inputs, outputs, compression_type);
}

Error KyTensorClient::AsyncInfer(OnCompleteFn callback, const InferOptions& options,
                                const std::vector<InferInput*>& inputs,
                                const std::vector<const InferRequestedOutput*>& outputs,
                                KYTENSOR_COMPRESSION_TYPE compression_type) {
    return pImpl->AsyncInfer(callback, options, inputs, outputs, compression_type);
}

Error KyTensorClient::StartStream(OnCompleteFn callback, bool enable_stats,
                                 uint32_t stream_timeout, KYTENSOR_COMPRESSION_TYPE compression_type) {
    return pImpl->StartStream(callback, enable_stats, stream_timeout, compression_type);
}

Error KyTensorClient::StopStream() {
    return pImpl->StopStream();
}

Error KyTensorClient::AsyncStreamInfer(const InferOptions& options, const std::vector<InferInput*>& inputs,
                                     const std::vector<const InferRequestedOutput*>& outputs) {
    return pImpl->AsyncStreamInfer(options, inputs, outputs);
}

}}