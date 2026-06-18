#pragma once
#include <string>

#include "log.h"
#include "triton/core/tritonserver.h"
#include "kytensor_service.pb.h"
#include "health.pb.h"

namespace kytensor { namespace server {

// The step of processing that the state is in. Every state must
// recognize START, COMPLETE and FINISH and the others are optional.
typedef enum {
    // This marks the starting stage of the RPC
    START,
    // This marks that RPC is complete.
    COMPLETE,
    // This marks the stage where all the notifications from the gRPC
    // completion queue is received and state can be safely released.
    FINISH,
    // This stage means that RPC has been issued to Triton for inference
    // and is waiting for the server callbacks or cancellation to be
    // invoked.
    ISSUED,
    // This stage means the request has been read from the network and
    // can be sent to Triton for execution.
    READ,
    // This stage means that the response is ready to be written back to
    // the network.
    WRITEREADY,
    // This stage means that response has been written completely to the
    // network.
    WRITTEN,
    // This marks the special stage for the state object to differentiate
    // the tag delivered from AsyncNotifyWhenDone() method.
    WAITING_NOTIFICATION,
    // This stage means that the cancellation for the RPC has been issued
    // to the server.
    CANCELLATION_ISSUED,
    // This stage marks that the state has been successfully cancelled.
    CANCELLED,
    // This is intermediary stage where the state has been been partially
    // completed by grpc responder Finish call or AsyncNotifyWhenDone()
    // notification. The other next call will move the stage to fully
    // complete.
    PARTIAL_COMPLETION
} Steps;

template <typename TensorType>
TRITONSERVER_Error*
ParseSharedMemoryParams(
    const TensorType& tensor, bool* has_shared_memory, std::string* region_name,
    int64_t* offset, size_t* byte_size)
{
    *has_shared_memory = false;
    *offset = 0 /* default value */;
    const auto& region_it = tensor.parameters().find("shared_memory_region");
    if (region_it != tensor.parameters().end()) {
        *has_shared_memory = true;
        const auto& infer_param = region_it->second;
        if (infer_param.parameter_choice_case() !=
            inference::InferParameter::ParameterChoiceCase::kStringParam) {
            return TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                std::string(
                    "invalid value type for 'shared_memory_region' parameter for "
                    "tensor '" +
                    tensor.name() + "', expected string_param.")
                    .c_str());
        }
        *region_name = infer_param.string_param();
    }

    const auto& offset_it = tensor.parameters().find("shared_memory_offset");
    if (offset_it != tensor.parameters().end()) {
        if (!*has_shared_memory) {
            return TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                std::string(
                    "'shared_memory_offset' can not be specified without "
                    "'shared_memory_region' parameter for tensor '" +
                    tensor.name() + "'")
                    .c_str());
        }
        const auto& infer_param = offset_it->second;
        if (infer_param.parameter_choice_case() !=
            inference::InferParameter::ParameterChoiceCase::kInt64Param) {
            return TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                std::string(
                    "invalid value type for 'shared_memory_offset' parameter for "
                    "tensor '" +
                    tensor.name() + "', expected int64_param.")
                    .c_str());
        }
        *offset = infer_param.int64_param();
    }

    const auto& bs_it = tensor.parameters().find("shared_memory_byte_size");
    if (bs_it != tensor.parameters().end()) {
        if (!*has_shared_memory) {
            return TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                std::string(
                    "'shared_memory_byte_size' can not be specified without "
                    "'shared_memory_region' parameter for tensor '" +
                    tensor.name() + "'")
                    .c_str());
        }
        const auto& infer_param = bs_it->second;
        if (infer_param.parameter_choice_case() !=
            inference::InferParameter::ParameterChoiceCase::kInt64Param) {
            return TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                std::string(
                    "invalid value type for 'shared_memory_byte_size' parameter "
                    "for "
                    "tensor '" +
                    tensor.name() + "', expected int64_param.")
                    .c_str());
        }
        *byte_size = infer_param.int64_param();
    } else {
        if (*has_shared_memory) {
            return TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                std::string(
                    "'shared_memory_byte_size' must be specified along with "
                    "'shared_memory_region' parameter for tensor '" +
                    tensor.name() + "'")
                    .c_str());
        }
    }

    return nullptr;
}

TRITONSERVER_Error* ParseClassificationParams(
    const inference::ModelInferRequest::InferRequestedOutputTensor& output,
    bool* has_classification, uint32_t* classification_count);

}}