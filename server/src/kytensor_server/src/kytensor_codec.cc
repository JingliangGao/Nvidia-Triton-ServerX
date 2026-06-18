#include "kytensor_codec.h"
#include "log.h"
#include <event2/buffer.h>
#include <event2/event.h>
#include <sys/time.h>

namespace kytensor { namespace server {

void KytensorCodec::ReceiveMessage(bufferevent *bev, void *arg)
{
    struct evbuffer *input = bufferevent_get_input(bev);
    size_t len = evbuffer_get_length(input);
    if (len < min_message_len) {

        LOGF_WRN("len < min_message_len: %ld < %d\n", len, min_message_len);
        return;
    }

    while (true) {
        uint32_t message_len = 0;
        uint32_t net_message_len = 0;
        MessageHeader header;
        evbuffer_copyout(input, &header, sizeof(MessageHeader));
        message_len = header.total_len;
        LOGF_DBG("len: %ld, msg header: total len: %d, msg id: %d, status: %d, name len:%d\n",
            len, header.total_len, header.msg_id, header.status, header.name_len);

            // invalid length
        if (message_len < min_message_len) {
            LOGF_DBG("invalid length: %d, len: %ld\n", message_len, len);
            break;
        }

        if (message_len > max_message_len) {
            LOGF_ERR("error length, message_len: %d\n", message_len);
            evbuffer_drain(input, evbuffer_get_length(input));
            break;
        }

        // incomplete data
        if (len < message_len || len < min_message_len) {
            LOGF_DBG("incomplete data, len < message_len: %ld < %d\n", len, message_len);
            //evbuffer_drain(input, evbuffer_get_length(input));
            break;
        }

        // Use buffer pool to reduce dynamic allocations
        if (buffer_.size() < message_len) {
            buffer_.resize(message_len);
        }

        int ev_len = evbuffer_remove(input, buffer_.data(), message_len);
        if ((uint32_t)ev_len != message_len) {
            // read error
            evbuffer_drain(input, evbuffer_get_length(input));
            break;
        }
        len -= message_len;

        ErrorCode error_code = NO_ERROR;
        MessagePtr message = ParseMessage(buffer_.data(), header, &error_code);
        if (error_code == NO_ERROR && message) {
            // Use microsecond precision for timestamp
            struct timeval tv;
            gettimeofday(&tv, nullptr);
            int64_t timestamp = tv.tv_sec * 1000000LL + tv.tv_usec;
            message_callback_(bev, header, message, timestamp);
        } else {
            LOG_ERR("%s: ParseMessage error, code: %d\n", __func__, error_code);
        }
    }
}

void KytensorCodec::ServerReceiveMessage(std::shared_ptr<Connection>& conn, void *arg)
{
    (void)arg;
    if (!conn || conn->is_closed()) {
        return;
    }
    struct bufferevent* bev = conn->bev();

    struct evbuffer *input = bufferevent_get_input(bev);
    size_t len = evbuffer_get_length(input);
    if (len < min_message_len) {

        LOGF_WRN("len < min_message_len: %ld < %d\n", len, min_message_len);
        return;
    }

    while (true) {
        uint32_t message_len = 0;
        uint32_t net_message_len = 0;
        MessageHeader header;
        evbuffer_copyout(input, &header, sizeof(MessageHeader));
        message_len = header.total_len;
        LOGF_DBG("len: %ld, msg header: total len: %d, msg id: %d, status: %d, name len:%d\n",
            len, header.total_len, header.msg_id, header.status, header.name_len);

         // invalid length
        if (message_len < min_message_len) {
            LOGF_DBG("invalid length: %d, len: %ld\n", message_len, len);
            break;
        }

        if (message_len > max_message_len) {
            LOGF_ERR("error length, message_len: %d\n", message_len);
            evbuffer_drain(input, evbuffer_get_length(input));
            break;
        }

        // incomplete data
        if (len < message_len || len < min_message_len) {
            LOGF_DBG("incomplete data, len < message_len: %ld < %d\n", len, message_len);
            //evbuffer_drain(input, evbuffer_get_length(input));
            break;
        }

        // Use buffer pool to reduce dynamic allocations
        if (buffer_.size() < message_len) {
            buffer_.resize(message_len);
        }

        int ev_len = evbuffer_remove(input, buffer_.data(), message_len);
        if ((uint32_t)ev_len != message_len) {
            // read error
            evbuffer_drain(input, evbuffer_get_length(input));
            break;
        }
        len -= message_len;

        ErrorCode error_code = NO_ERROR;
        MessagePtr message = ParseMessage(buffer_.data(), header, &error_code);
        if (error_code == NO_ERROR && message) {
            // Use microsecond precision for timestamp
            struct timeval tv;
            gettimeofday(&tv, nullptr);
            int64_t timestamp = tv.tv_sec * 1000000LL + tv.tv_usec;
            message_callback_(bev, header, message, timestamp);
        } else {
            LOG_ERR("%s: ParseMessage error, code: %d\n", __func__, error_code);
        }
    }
}

void KytensorCodec::SendMessage(const bufferevent *bev, const MessageHeader& header, const google::protobuf::Message &message)
{
    std::string type_name = message.GetTypeName();
    uint32_t message_len = static_cast<uint32_t>(message.ByteSizeLong());
    MessageHeader write_header = header;
    write_header.name_len = static_cast<uint32_t>(type_name.size());
    write_header.total_len = sizeof(MessageHeader) + write_header.name_len + message_len;

    LOGF_DBG("msg header: total len: %d, msg id: %d, status: %d, name len:%d, bev:%p\n",
        write_header.total_len, write_header.msg_id, write_header.status, write_header.name_len, bev);

    if (bev == nullptr) {
        LOGF_ERR("send message handler is null\n");
        return;
    }

    // Lock the bufferevent to prevent concurrent writes from interleaving
    // the header/type_name/data evbuffer_add sequence.
    struct bufferevent* bev_mut = const_cast<struct bufferevent*>(bev);
    bufferevent_lock(bev_mut);

    struct evbuffer *output = bufferevent_get_output(bev_mut);
    if(!output) {
        LOGF_ERR("bufferevent_get_output failed\n");
        bufferevent_unlock(bev_mut);
        return;
    }

    if (evbuffer_add(output, &write_header, header_len_) != 0) {
        LOGF_ERR("bufferevent_write failed for message header\n");
        bufferevent_unlock(bev_mut);
        return;
    }

    if (evbuffer_add(output, type_name.data(), type_name.size()) != 0) {
        LOGF_ERR("bufferevent_write failed for type name\n");
        bufferevent_unlock(bev_mut);
        return;
    }

    if (evbuffer_add(output, message.SerializeAsString().data(), message.ByteSizeLong()) != 0) {
        LOG_ERR("bufferevent_write failed for message body\n");
        bufferevent_unlock(bev_mut);
        return;
    }

    if (bufferevent_flush(bev_mut, EV_WRITE, BEV_FLUSH) != 0) {
        LOGF_ERR("bufferevent_flush failed\n");
        bufferevent_unlock(bev_mut);
        return;
    }

    bufferevent_unlock(bev_mut);
}

void KytensorCodec::ServerSendMessage(const std::shared_ptr<Connection>& conn, const MessageHeader& header, const google::protobuf::Message &message)
{
    if (!conn || conn->is_closed()) {
        return;
    }
    struct bufferevent* bev = conn->bev();

    std::string type_name = message.GetTypeName();
    uint32_t message_len = static_cast<uint32_t>(message.ByteSizeLong());
    MessageHeader write_header = header;
    write_header.name_len = static_cast<uint32_t>(type_name.size());
    write_header.total_len = sizeof(MessageHeader) + write_header.name_len + message_len;

    LOGF_DBG("msg header: total len: %d, msg id: %d, status: %d, name len:%d, bev:%p\n",
        write_header.total_len, write_header.msg_id, write_header.status, write_header.name_len, bev);

    // Lock the bufferevent to prevent concurrent writes from interleaving
    // the header/type_name/data evbuffer_add sequence.
    bufferevent_lock(bev);

    struct evbuffer *output = bufferevent_get_output(bev);
    if(!output) {
        LOGF_ERR("bufferevent_get_output failed\n");
        bufferevent_unlock(bev);
        return;
    }

    if (evbuffer_add(output, &write_header, header_len_) != 0) {
        LOGF_ERR("bufferevent_write failed for message header\n");
        bufferevent_unlock(bev);
        return;
    }

    if (evbuffer_add(output, type_name.data(), type_name.size()) != 0) {
        LOGF_ERR("bufferevent_write failed for type name\n");
        bufferevent_unlock(bev);
        return;
    }

    if (evbuffer_add(output, message.SerializeAsString().data(), message.ByteSizeLong()) != 0) {
        LOG_ERR("bufferevent_write failed for message body\n");
        bufferevent_unlock(bev);
        return;
    }

    if (bufferevent_flush(bev, EV_WRITE, BEV_FLUSH) != 0) {
        LOGF_ERR("bufferevent_flush failed\n");
        bufferevent_unlock(bev);
        return;
    }

    bufferevent_unlock(bev);
}

google::protobuf::Message *KytensorCodec::CreateMessageByType(const std::string &type_name)
{
    google::protobuf::Message* message = nullptr;
    const google::protobuf::Descriptor* descriptor =
        google::protobuf::DescriptorPool::generated_pool()->FindMessageTypeByName(type_name);
    if (descriptor) {
        const google::protobuf::Message* prototype =
            google::protobuf::MessageFactory::generated_factory()->GetPrototype(descriptor);
        if (prototype) {
            message = prototype->New();
        }
    }
    return message;
}

MessagePtr KytensorCodec::ParseMessage(const char* buf, const MessageHeader& header, ErrorCode* error_code)
{
    MessagePtr message;

    //protobuf data
    std::string type_name(buf + header_len_, header.name_len);
    message.reset(CreateMessageByType(type_name));
    LOGF_DBG("type_name: %s\n", type_name.c_str());
    if (message) {
        const char* data = buf + header_len_ + header.name_len;
        int data_len = header.total_len - header_len_ - header.name_len;
        if (message->ParseFromArray(data, data_len)) {
            *error_code = NO_ERROR;
        } else {
            *error_code = DECODE_FAILURE;
        }
    } else {
        *error_code = UNKNOWN_TYPE;
    }

    return message;
}

}}
