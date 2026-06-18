#include "utils.h"
#include <string>
#include <sstream>

namespace triton::backend::llamacpp::utils
{

std::string get_request_prompt(TRITONBACKEND_Request* request)
{
    uint32_t num_inputs;
    std::string ret;
    LOG_IF_ERROR(TRITONBACKEND_RequestInputCount(request, &num_inputs), "Error getting input count");

    for (uint32_t i = 0; i < num_inputs; i++) {
        TRITONBACKEND_Input* input = nullptr;
        LOG_IF_ERROR(TRITONBACKEND_RequestInputByIndex(request, i, &input), "Error getting input index");

        char const* input_name = nullptr;
        TRITONSERVER_DataType data_type = TRITONSERVER_TYPE_INVALID;
        int64_t const* shape = nullptr;
        uint32_t dims_count = 0;
        uint64_t byte_size = 0;
        uint32_t buffer_count = 0;
        LOG_IF_ERROR(TRITONBACKEND_InputProperties(
                         input, &input_name, &data_type, &shape, &dims_count, &byte_size, &buffer_count),
            "Error getting input properties");

        uint64_t buffer_offset = 0;
        std::vector<char> buf(byte_size);
        for (int64_t buffer_id = 0; buffer_id < buffer_count; buffer_id++) {
            void const* buffer = nullptr;
            uint64_t buffer_byte_size = 0;
            TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
            int64_t memory_type_id = 0;
            LOG_IF_ERROR(
                TRITONBACKEND_InputBuffer(input, buffer_id, &buffer, &buffer_byte_size, &memory_type, &memory_type_id),
                "failed to get input buffer");

            std::memcpy(buf.data() + buffer_offset, buffer, buffer_byte_size);
            buffer_offset += buffer_byte_size;
        }

        std::string res(buf.data()+4, byte_size-4);
        ret = res;
    }

    return ret;
}

std::string get_request_prompt(TRITONBACKEND_Request* request, std::string const& input_name)
{
    TRITONBACKEND_Input *input;
    TRITONSERVER_Error *error = TRITONBACKEND_RequestInput(request, input_name.c_str(), &input);

    if (error) {
        std::string msg
            = "did not not provide "
            + input_name + " input for the request";
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, msg.c_str());
        TRITONSERVER_ErrorDelete(error);
        return "";
    }

    uint64_t input_byte_size = 0;
    uint32_t buffer_count = 0;
    TRITONBACKEND_InputProperties(input, nullptr, nullptr, nullptr, nullptr, &input_byte_size, &buffer_count);

    void const* buffer = 0L;
    uint64_t buffer_byte_size = 0;
    TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t memory_type_id = 0;
    TRITONBACKEND_InputBuffer(input, 0, &buffer, &buffer_byte_size, &memory_type, &memory_type_id);

    //first 4 bytes is len of string
    //mutil copy to be optimized
    std::vector<char> buf(input_byte_size);
    std::memcpy(buf.data(), buffer, buffer_byte_size);
    std::string res(buf.data()+4, buffer_byte_size-4);
    
    return res;
}

std::vector<float> get_request_lora_scale(TRITONBACKEND_Request* request, std::string const& input_name)
{
    TRITONBACKEND_Input *input;
    TRITONSERVER_Error *error = TRITONBACKEND_RequestInput(request, input_name.c_str(), &input);

    if (error) {
        std::string msg
            = "did not not provide "
            + input_name + " input for the request";
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, msg.c_str());
        TRITONSERVER_ErrorDelete(error);
        return std::vector<float>();
    }

    uint64_t input_byte_size = 0;
    uint32_t buffer_count = 0;
    TRITONBACKEND_InputProperties(input, nullptr, nullptr, nullptr, nullptr, &input_byte_size, &buffer_count);

    void const* buffer = 0L;
    uint64_t buffer_byte_size = 0;
    TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t memory_type_id = 0;
    TRITONBACKEND_InputBuffer(input, 0, &buffer, &buffer_byte_size, &memory_type, &memory_type_id);

    //first 4 bytes is len of string
    //mutil copy to be optimized
    //std::vector<char> buf(input_byte_size);
    //std::memcpy(buf.data(), buffer, buffer_byte_size);
    //std::string res(buf.data()+4, buffer_byte_size-4);
    std::vector<float> buf(input_byte_size/sizeof(float));
    std::memcpy(buf.data(), buffer, buffer_byte_size);
    
    return buf;
}

std::vector<std::string> split(std::string const& str, char delimiter)
{
    std::vector<std::string> tokens;
    std::string token;
    std::stringstream ss(str);

    while (std::getline(ss, token, delimiter))
    {
        tokens.push_back(token);
    }

    return tokens;
}

#if 0
int get_request_input(TRITONBACKEND_Request* request, const std::string& input_name)
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
        return -1;
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

    int ret = *reinterpret_cast<const int*>(buffer);

    return ret;
}
#endif
}