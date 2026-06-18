#include "triton/backend/backend_common.h"
#include "triton/core/tritonbackend.h"
#include "triton/core/tritonserver.h"

namespace triton::backend::llamacpp
{

namespace utils
{

std::string get_request_prompt(TRITONBACKEND_Request* request);
std::string get_request_prompt(TRITONBACKEND_Request* request, std::string const& input_name);
std::vector<std::string> split(std::string const& str, char delimiter);
std::vector<float> get_request_lora_scale(TRITONBACKEND_Request* request, std::string const& input_name);
//int get_request_input(TRITONBACKEND_Request* request, const std::string& input_name);

/// @brief Query Triton for a buffer that can be used to pass the output tensors
template <typename T>
void* getResponseBuffer(TRITONBACKEND_Response* tritonResponse, std::vector<int64_t> const& shape,
    TRITONSERVER_DataType dtype, std::string const& name)
{
    TRITONBACKEND_Output* output;
    TRITONSERVER_Error* err{nullptr};
    if (dtype == TRITONSERVER_TYPE_BYTES) {
        std::vector<int64_t> bytes_shape{1, 1};
        err = TRITONBACKEND_ResponseOutput(tritonResponse, &output, name.c_str(), dtype, bytes_shape.data(), bytes_shape.size());
    } else {
        err = TRITONBACKEND_ResponseOutput(tritonResponse, &output, name.c_str(), dtype, shape.data(), shape.size());
    }

    if (err != nullptr)
    {
        auto errMsg = TRITONSERVER_ErrorMessage(err);
        //TLLM_THROW("Could not get response output for output tensor %s: %s", name.c_str(), errMsg);
        LOG_MESSAGE(
        TRITONSERVER_LOG_ERROR,
        (std::string("Could not get response output for output tensor " + name + ": ") 
        + std::string(errMsg)).c_str());
    }

    TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t memory_type_id = 0;
    uint64_t size = 1;
    for (auto s : shape)
    {
        size *= s;
    }
    auto buffersize = size * sizeof(T);
    void* tritonBuffer = 0L;
    err = TRITONBACKEND_OutputBuffer(output, &tritonBuffer, buffersize, &memory_type, &memory_type_id);
    if (err != nullptr)
    {
        auto errMsg = TRITONSERVER_ErrorMessage(err);
        //TLLM_THROW("Could not get output buffer for output tensor %s: %s", name.c_str(), errMsg);
        LOG_MESSAGE(
        TRITONSERVER_LOG_ERROR,
        (std::string("Could not get response output for output tensor " + name + ": ") 
        + std::string(errMsg)).c_str());
    }
    return tritonBuffer;
}

template <typename T>
int get_request_input(TRITONBACKEND_Request* request, const std::string& input_name, T& value)
{
    // Get stop signal from the request
    TRITONBACKEND_Input* input;
    TRITONSERVER_Error* error = TRITONBACKEND_RequestInput(request, input_name.c_str(), &input);
    if (error)
    {
        // If the user does not provide input "stop", then regard the request as
        // unstopped
        std::string msg
            = "ModelInstanceState::getRequestBooleanInputTensor: user "
              "did not not provide "
            + input_name + " input for the request";
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, msg.c_str());
        TRITONSERVER_ErrorDelete(error);
        return 0;
    }

    uint64_t input_byte_size = 0;
    uint32_t buffer_count = 0;
    TRITONBACKEND_InputProperties(input, nullptr, nullptr, nullptr, nullptr, &input_byte_size, &buffer_count);

    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
        ("ModelInstanceState::getRequestStopSignal: buffer_count = " + std::to_string(buffer_count)).c_str());

    const void* buffer = 0L;
    uint64_t buffer_byte_size = 0;
    TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t memory_type_id = 0;
    TRITONBACKEND_InputBuffer(input, 0, &buffer, &buffer_byte_size, &memory_type, &memory_type_id);

    assert((memory_type == TRITONSERVER_MEMORY_CPU) || (memory_type == TRITONSERVER_MEMORY_CPU_PINNED));

    value = *reinterpret_cast<const T*>(buffer);

    return 1;
}

}

}