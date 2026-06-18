#pragma once
#include <cstdint>
#include <event2/bufferevent.h>
#include <vector>
#include <memory>

#include "kytensor_common.h"

namespace kytensor { namespace server {

enum MessageType {
    MESSAGE_TYPE_UNKNOWN = 0,
    MESSAGE_TYPE_REQUEST = 1,
    MESSAGE_TYPE_RESPONSE = 2,
};

enum MessageInferType {
    MESSAGE_INFER_COMMON = 0,
    MESSAGE_INFER_SYNC,
    MESSAGE_INFER_ASYNC,
    MESSAGE_INFER_STREAM,
};

// message protocol
//  KytensorProtocolFormat
//     uint32_t len;
//     uint32_t msg_id;
//     uint32_t name_len;
//     uint8_t status_code;
//     char type_name[name_len];
//     char protobuf_data[len - name_len - sizeof(MessageHeader)];
struct MessageHeader {
    uint32_t total_len;
    uint32_t msg_id;
    uint8_t msg_type:4;
    uint8_t msg_infer_type:4;
    uint8_t status;
    uint8_t version;
    uint8_t reserved;
    uint32_t name_len;
};

class KytensorCodec {

public:
    enum ErrorCode {
        NO_ERROR = 0,
        INVALID_LENGTH = 1,
        INVALID_NAME_LENGTH = 2,
        INCOMPLETE_DATA = 3,
        UNKNOWN_TYPE = 4,
        DECODE_FAILURE = 5
    };

    typedef std::function <void (const struct bufferevent*,
                                    const MessageHeader& header,
                                    MessagePtr&,
                                    int64_t)> MessageCallback;

    KytensorCodec(const MessageCallback& cb)
        : message_callback_(cb), buffer_(BUFFER_POOL_SIZE) {

    }

    void ReceiveMessage(bufferevent *bev, void *arg);
    void ServerReceiveMessage(std::shared_ptr<Connection>& conn, void *arg);
    void SendMessage(const bufferevent *bev, const MessageHeader& header, const google::protobuf::Message &message);
    void ServerSendMessage(const std::shared_ptr<Connection>& conn, const MessageHeader& header, const google::protobuf::Message &message);

    static google::protobuf::Message* CreateMessageByType(const std::string& type_name);
    static MessagePtr ParseMessage(const char* buf, const MessageHeader& header, ErrorCode* error_code);

private:
    MessageCallback message_callback_;

    // Buffer for message processing to reduce dynamic allocations
    mutable std::vector<char> buffer_;
    static constexpr size_t BUFFER_POOL_SIZE = 8192;

    const static int header_len_ = sizeof(MessageHeader);
    const static int min_message_len = sizeof(MessageHeader) + 2;
    const static int max_message_len = 64 * 1024 * 1024;

};

}}