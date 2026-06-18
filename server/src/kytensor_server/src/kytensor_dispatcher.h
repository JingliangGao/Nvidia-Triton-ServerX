#pragma once
#include <map>

#include "kytensor_common.h"

namespace kytensor { namespace server {

class Callback {

public:
    virtual ~Callback() = default;
    virtual void RecieveMessage(const struct bufferevent* bev,
                                const MessageHeader& header,
                                const MessagePtr& message,
                                int64_t receive_time) const = 0;
};

template <typename T>
class CallbackT : public Callback {

    static_assert(std::is_base_of<google::protobuf::Message, T>::value,
                  "T must be derived from google::protobuf::Message");
public:
    typedef std::function<void (const struct bufferevent*,  const MessageHeader&, const std::shared_ptr<T>&, int64_t)> MessageTCallback;

    CallbackT(const MessageTCallback& cb)
        : callback_(cb) {

    }

    void RecieveMessage(const struct bufferevent* bev,
                        const MessageHeader& header,
                        const MessagePtr& message,
                        int64_t receive_time) const override {
        std::shared_ptr<T> typed_msg = std::dynamic_pointer_cast<T>(message);
        assert(typed_msg != nullptr);
        callback_(bev, header, typed_msg, receive_time);
    }

private:
    MessageTCallback callback_;
};

class KytensorDispatcher {

public:
    typedef std::function<void (const struct bufferevent*,
                                const MessagePtr& message,
                                    int64_t)> KytensorMessageCallback;

    KytensorDispatcher(const KytensorMessageCallback& cb)
        : default_callback_(cb) {

    }

    void RecieveKytensorMessage(const struct bufferevent* bev,
                            const MessageHeader& header,
                           const MessagePtr& message,
                           int64_t receive_time) const {
        auto it = callback_map_.find(message->GetDescriptor());
        if (it != callback_map_.end()) {
            it->second->RecieveMessage(bev, header, message, receive_time);
        } else if (default_callback_) {
            default_callback_(bev, message, receive_time);
        }
    }

    template<typename T>
    void RegisterMessageCallback(const typename CallbackT<T>::MessageTCallback& cb) {
        std::shared_ptr<CallbackT<T>> pd(new CallbackT<T>(cb));
        callback_map_[T::descriptor()] = pd;
    }

private:
    typedef std::map<const google::protobuf::Descriptor*, std::shared_ptr<Callback>> CallbackMap;
    CallbackMap callback_map_;
    KytensorMessageCallback default_callback_;
};

}}