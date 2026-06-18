#include "kytensor_common.h"
#include "log.h"

namespace kytensor { namespace server {

TRITONSERVER_Error*
GetModelVersionFromString(const std::string& version_string, int64_t* version)
{
    if (version_string.empty()) {
    *version = -1;
    return nullptr;  // success
    }

    try {
    *version = std::stol(version_string);
    }
    catch (std::exception& e) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string(
            "failed to get model version from specified version string '" +
            version_string + "' (details: " + e.what() +
            "), version should be an integral value > 0")
            .c_str());
    }

    if (*version < 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string(
            "invalid model version specified '" + version_string +
            "' , version should be an integral value > 0")
            .c_str());
    }

    return nullptr;  // success
}

TRITONSERVER_Error*
SetStateParameterFromTritonParameter(
    StateParameters& state_params,
    const std::pair<std::string, inference::InferParameter>& param)
{
    const auto& key = param.first;
    const auto& value = param.second;
    if (key == "triton_enable_empty_final_response") {
        if (value.parameter_choice_case() !=
            inference::InferParameter::ParameterChoiceCase::kBoolParam) {
            return TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                (std::string("invalid value type for '") + key +
                std::string("' parameter, expected bool_param."))
                    .c_str());
        }
        state_params.enable_empty_final_response_ = value.bool_param();
    }

    return nullptr;  // success
}

TRITONSERVER_Error*
OutputBufferAttributesHelper(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    const TensorShmMap& shm_map,
    TRITONSERVER_BufferAttributes* buffer_attributes)
{
    // We only need to set the cuda ipc handle here. The rest of the buffer
    // attributes have been properly populated by triton core.
    if (tensor_name != nullptr) {
        const auto& pr = shm_map.find(tensor_name);

        if (pr != shm_map.end()) {
            if (pr->second.memory_type_ == TRITONSERVER_MEMORY_GPU) {
                RETURN_IF_ERR(TRITONSERVER_BufferAttributesSetCudaIpcHandle(
                    buffer_attributes, pr->second.cuda_ipc_handle_));
            }
        }
    }

    return nullptr;  // Success
}

TRITONSERVER_Error*
OutputBufferQueryHelper(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t* byte_size, const TensorShmMap& shm_map,
    TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id)
{
    // Check if shared memory is used if named tensor is provided
    if (tensor_name != nullptr) {
        const auto& pr = shm_map.find(tensor_name);
        if (pr != shm_map.end()) {
            // The output is in shared memory so check that shared memory
            // size is at least large enough for the output, if byte size is provided
            if ((byte_size != nullptr) && (*byte_size > pr->second.byte_size_)) {
                // Don't return error yet and just set to the default properties for
                // GRPC buffer, error will be raised when allocation happens
                *memory_type = TRITONSERVER_MEMORY_CPU;
                *memory_type_id = 0;
            } else {
                *memory_type = pr->second.memory_type_;
                *memory_type_id = pr->second.memory_type_id_;
            }
            return nullptr;  // Success
        }
    }

    // Not using shared memory so a buffer created directly in
    // the response protobuf will be used, and the type will be CPU.
    *memory_type = TRITONSERVER_MEMORY_CPU;
    *memory_type_id = 0;
    return nullptr;  // Success
}

TRITONSERVER_Error*
ResponseAllocatorHelper(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, inference::ModelInferResponse* response,
    const TensorShmMap& shm_map, void** buffer, void** buffer_userp,
    TRITONSERVER_MemoryType* actual_memory_type, int64_t* actual_memory_type_id)
{
    *buffer = nullptr;
    *buffer_userp = nullptr;
    *actual_memory_type = preferred_memory_type;
    *actual_memory_type_id = preferred_memory_type_id;

    // We add an output contents even if the 'byte_size' == 0 because we
    // expect to have a contents for every output.
    inference::ModelInferResponse::InferOutputTensor* output_tensor =
        response->add_outputs();
    output_tensor->set_name(tensor_name);
    std::string* raw_output = response->add_raw_output_contents();
    //LOGF_INF("output size: %d\n", response->outputs_size());
    if (byte_size > 0) {
        const auto& pr = shm_map.find(tensor_name);
        if (pr != shm_map.end()) {
            // The output is in shared memory so check that shared memory
            // size is at least large enough for the output.
            if (byte_size > pr->second.byte_size_) {
                return TRITONSERVER_ErrorNew(
                    TRITONSERVER_ERROR_INTERNAL,
                    std::string(
                        "shared memory size specified with the request for output '" +
                        std::string(tensor_name) + "' (" +
                        std::to_string(pr->second.byte_size_) +
                        " bytes) should be at least " + std::to_string(byte_size) +
                        " bytes to hold the results")
                        .c_str());
            }

            *buffer = const_cast<void*>(pr->second.base_);
            *actual_memory_type = pr->second.memory_type_;
            *actual_memory_type_id = pr->second.memory_type_id_;

            LOGF_DBG("Kytensor: using shared-memory for %s, size: %ld, addr: %p\n", tensor_name, byte_size, *buffer);
        }

        // Not using shared memory so allocate a buffer. The buffer we
        // create is directly in the response protobuf so we can't
        // allocate any type other than CPU.
        //
        // FIXME we could use pinned CPU memory here.
        if (*actual_memory_type != TRITONSERVER_MEMORY_CPU) {
            LOGF_DBG("Kytensor: unable to provide '%s' in %s, will use CPU",
                            tensor_name,
                            TRITONSERVER_MemoryTypeString(*actual_memory_type));
            *actual_memory_type = TRITONSERVER_MEMORY_CPU;
            *actual_memory_type_id = 0;
        }

        raw_output->resize(byte_size);
        *buffer = static_cast<void*>(&((*raw_output)[0]));

        LOGF_DBG("Kytensor: using buffer for %s, size: %ld, addr: %p\n", tensor_name, byte_size, *buffer);
    }

    return nullptr;  // Success
}

TRITONSERVER_Error*
SetInferenceRequestMetadata(
    TRITONSERVER_InferenceRequest* inference_request,
    const inference::ModelInferRequest& request, StateParameters& state_params)
{
    RETURN_IF_ERR(TRITONSERVER_InferenceRequestSetId(
        inference_request, request.id().c_str()));

    uint32_t flags = 0;
    for (auto param : request.parameters()) {
        if (param.first.compare("sequence_id") == 0) {
            const auto& infer_param = param.second;
            if (infer_param.parameter_choice_case() ==
                inference::InferParameter::ParameterChoiceCase::kInt64Param) {
                RETURN_IF_ERR(TRITONSERVER_InferenceRequestSetCorrelationId(
                    inference_request, infer_param.int64_param()));
            } else if (
                infer_param.parameter_choice_case() ==
                inference::InferParameter::ParameterChoiceCase::kStringParam) {
                RETURN_IF_ERR(TRITONSERVER_InferenceRequestSetCorrelationIdString(
                    inference_request, infer_param.string_param().c_str()));
            } else {
                return TRITONSERVER_ErrorNew(
                    TRITONSERVER_ERROR_INVALID_ARG,
                    "invalid value type for 'sequence_id' parameter, expected "
                    "int64_param or string_param.");
            }
        } else if (param.first.compare("sequence_start") == 0) {
            const auto& infer_param = param.second;
            if (infer_param.parameter_choice_case() !=
                inference::InferParameter::ParameterChoiceCase::kBoolParam) {
                return TRITONSERVER_ErrorNew(
                    TRITONSERVER_ERROR_INVALID_ARG,
                    "invalid value type for 'sequence_start' parameter, expected "
                    "bool_param.");
        }
        if (infer_param.bool_param()) {
            flags |= TRITONSERVER_REQUEST_FLAG_SEQUENCE_START;
        }
        } else if (param.first.compare("sequence_end") == 0) {
            const auto& infer_param = param.second;
            if (infer_param.parameter_choice_case() !=
                inference::InferParameter::ParameterChoiceCase::kBoolParam) {
                return TRITONSERVER_ErrorNew(
                    TRITONSERVER_ERROR_INVALID_ARG,
                    "invalid value type for 'sequence_end' parameter, expected "
                    "bool_param.");
            }
        if (infer_param.bool_param()) {
            flags |= TRITONSERVER_REQUEST_FLAG_SEQUENCE_END;
        }
        } else if (param.first.compare("priority") == 0) {
            const auto& infer_param = param.second;
            if (infer_param.parameter_choice_case() ==
                inference::InferParameter::ParameterChoiceCase::kInt64Param) {
                if (infer_param.int64_param() < 0) {
                return TRITONSERVER_ErrorNew(
                    TRITONSERVER_ERROR_INVALID_ARG,
                    "invalid value for 'priority', expected value >= 0.");
                }
                RETURN_IF_ERR(TRITONSERVER_InferenceRequestSetPriorityUInt64(
                    inference_request, infer_param.int64_param()));
            } else if (
                infer_param.parameter_choice_case() ==
                inference::InferParameter::ParameterChoiceCase::kUint64Param) {
                RETURN_IF_ERR(TRITONSERVER_InferenceRequestSetPriorityUInt64(
                    inference_request, infer_param.uint64_param()));
            } else {
                return TRITONSERVER_ErrorNew(
                    TRITONSERVER_ERROR_INVALID_ARG,
                    "invalid value type for 'priority' parameter, expected "
                    "int64_param or uint64_param.");
            }
        } else if (param.first.compare("timeout") == 0) {
            const auto& infer_param = param.second;
            if (infer_param.parameter_choice_case() !=
                inference::InferParameter::ParameterChoiceCase::kInt64Param) {
                return TRITONSERVER_ErrorNew(
                    TRITONSERVER_ERROR_INVALID_ARG,
                    "invalid value type for 'timeout' parameter, expected "
                    "int64_param.");
            }
            RETURN_IF_ERR(TRITONSERVER_InferenceRequestSetTimeoutMicroseconds(
                inference_request, infer_param.int64_param()));
        } else if (param.first.rfind("triton_", 0) == 0) {
            if (!triton::server::Contains(triton::server::TRITON_RESERVED_REQUEST_PARAMS, param.first)) {
                return TRITONSERVER_ErrorNew(
                    TRITONSERVER_ERROR_INVALID_ARG,
                    (std::string(
                        "parameter keys starting with 'triton_' are reserved for "
                        "Triton "
                        "usage. Only the following keys starting with 'triton_' are "
                        "allowed: ") +
                    triton::server::Join(triton::server::TRITON_RESERVED_REQUEST_PARAMS, " "))
                        .c_str());
            }
            RETURN_IF_ERR(SetStateParameterFromTritonParameter(state_params, param));
        } else {
            const auto& infer_param = param.second;
            if (infer_param.parameter_choice_case() ==
                inference::InferParameter::ParameterChoiceCase::kInt64Param) {
                RETURN_IF_ERR(TRITONSERVER_InferenceRequestSetIntParameter(
                    inference_request, param.first.c_str(), infer_param.int64_param()));
            } else if (
                infer_param.parameter_choice_case() ==
                inference::InferParameter::ParameterChoiceCase::kBoolParam) {
                RETURN_IF_ERR(TRITONSERVER_InferenceRequestSetBoolParameter(
                    inference_request, param.first.c_str(), infer_param.bool_param()));
            } else if (
                infer_param.parameter_choice_case() ==
                inference::InferParameter::ParameterChoiceCase::kStringParam) {
                RETURN_IF_ERR(TRITONSERVER_InferenceRequestSetStringParameter(
                    inference_request, param.first.c_str(),
                    infer_param.string_param().c_str()));
            } else if (
                infer_param.parameter_choice_case() ==
                inference::InferParameter::ParameterChoiceCase::kDoubleParam) {
                RETURN_IF_ERR(TRITONSERVER_InferenceRequestSetDoubleParameter(
                    inference_request, param.first.c_str(),
                    infer_param.double_param()));
            } else {
                return TRITONSERVER_ErrorNew(
                    TRITONSERVER_ERROR_INVALID_ARG,
                    std::string(
                        "invalid value type for '" + param.first +
                        "' parameter, expected "
                        "int64_param, bool_param, or string_param.")
                        .c_str());
            }
        }
    }

    RETURN_IF_ERR(
        TRITONSERVER_InferenceRequestSetFlags(inference_request, flags));

    for (const auto& input : request.inputs()) {
        RETURN_IF_ERR(TRITONSERVER_InferenceRequestAddInput(
            inference_request, input.name().c_str(),
            TRITONSERVER_StringToDataType(input.datatype().c_str()),
            input.shape().data(), input.shape_size()));
    }

    for (const auto& output : request.outputs()) {
        RETURN_IF_ERR(TRITONSERVER_InferenceRequestAddRequestedOutput(
            inference_request, output.name().c_str()));
    }

    return nullptr;  // Success
}

TRITONSERVER_Error*
InferKytensorToInputHelper(
    const std::string& input_name, const std::string& model_name,
    const TRITONSERVER_DataType tensor_dt, const TRITONSERVER_DataType input_dt,
    const size_t binary_data_byte_size)
{
    if (binary_data_byte_size != 0) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            std::string(
                "unexpected explicit tensor data for input tensor '" + input_name +
                "' for model '" + model_name +
                "', binary data was already supplied.")
                .c_str());
    }

    if (tensor_dt != input_dt) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            std::string(
                "unexpected explicit tensor data for input tensor '" + input_name +
                "' for model '" + model_name + "' of type '" +
                TRITONSERVER_DataTypeString(tensor_dt) + "', expected datatype '" +
                TRITONSERVER_DataTypeString(input_dt) + "'")
                .c_str());
    }

    return nullptr;  // success
}

TRITONSERVER_Error*
InferKytensorToInput(
    const std::shared_ptr<TRITONSERVER_Server>& tritonserver,
    const std::shared_ptr<triton::server::SharedMemoryManager>& shm_manager,
    const inference::ModelInferRequest& request,
    std::list<std::string>* serialized_data,
    TRITONSERVER_InferenceRequest* inference_request)
{
    // Verify that the batch-byte-size of each input matches the size of
    // the provided tensor data (provided raw or from shared memory)
    int index = 0;
    for (const auto& io : request.inputs()) {
        const void* base;
        size_t byte_size = 0;
        TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
        int64_t memory_type_id = 0;

        std::string region_name;
        int64_t offset;
        bool has_shared_memory;
        RETURN_IF_ERR(
            ParseSharedMemoryParams<inference::ModelInferRequest::InferInputTensor>(
                io, &has_shared_memory, &region_name, &offset, &byte_size));

        TRITONSERVER_BufferAttributes* buffer_attributes;
        RETURN_IF_ERR(TRITONSERVER_BufferAttributesNew(&buffer_attributes));
        auto buffer_attributes_del =
            [](TRITONSERVER_BufferAttributes* buffer_attributes) {
            TRITONSERVER_BufferAttributesDelete(buffer_attributes);
            };
        std::unique_ptr<
            TRITONSERVER_BufferAttributes, decltype(buffer_attributes_del)>
            buffer_attrsl(buffer_attributes, buffer_attributes_del);
        char* cuda_ipc_handle = nullptr;

        if (has_shared_memory) {
            if (io.has_contents()) {
                return TRITONSERVER_ErrorNew(
                    TRITONSERVER_ERROR_INVALID_ARG,
                    std::string(
                        "unexpected 'content' provided when using shared memory "
                        "for "
                        "input tensor '" +
                        io.name() + "' for model '" + request.model_name() + "'")
                        .c_str());
            }
            void* tmp;
            RETURN_IF_ERR(shm_manager->GetMemoryInfo(
                region_name, offset, byte_size, &tmp, &memory_type, &memory_type_id));
            base = tmp;
            if (memory_type == TRITONSERVER_MEMORY_GPU) {
        #ifdef TRITON_ENABLE_GPU
                RETURN_IF_ERR(shm_manager->GetCUDAHandle(
                    region_name,
                    reinterpret_cast<cudaIpcMemHandle_t**>(&cuda_ipc_handle)));
        #endif
            }
        } else {
            if (io.has_contents() && (!request.raw_input_contents().empty())) {
                return TRITONSERVER_ErrorNew(
                    TRITONSERVER_ERROR_INVALID_ARG,
                    std::string(
                        "contents field must not be specified when using "
                        "raw_input_contents for '" +
                        io.name() + "' for model '" + request.model_name() + "'")
                        .c_str());
            } else if (io.has_contents()) {
                // Check the presence of explicit tensors
                TRITONSERVER_DataType dtype =
                    TRITONSERVER_StringToDataType(io.datatype().c_str());
                const size_t elem_byte_size = TRITONSERVER_DataTypeByteSize(dtype);
                if (io.contents().bool_contents_size() != 0) {
                    RETURN_IF_ERR(InferKytensorToInputHelper(
                        io.name(), request.model_name(), TRITONSERVER_TYPE_BOOL, dtype,
                        byte_size));
                    base = (const void*)io.contents().bool_contents().data();
                    byte_size = io.contents().bool_contents_size() * elem_byte_size;
                }

                if (io.contents().int_contents_size() != 0) {
                    if (dtype == TRITONSERVER_TYPE_INT8) {
                        RETURN_IF_ERR(InferKytensorToInputHelper(
                            io.name(), request.model_name(), TRITONSERVER_TYPE_INT8, dtype,
                            byte_size));
                        serialized_data->emplace_back();
                        auto& serialized = serialized_data->back();
                        serialized.reserve(
                            io.contents().int_contents_size() * elem_byte_size);
                        for (const auto& element : io.contents().int_contents()) {
                            // Assuming the system is little-endian, picking the
                            // least significant byte of 32-bit integer as a
                            // int8 element
                            serialized.append(
                                reinterpret_cast<const char*>(&element), elem_byte_size);
                        }
                        base = serialized.c_str();
                        byte_size = serialized.size();
                    } else if (dtype == TRITONSERVER_TYPE_INT16) {
                        RETURN_IF_ERR(InferKytensorToInputHelper(
                            io.name(), request.model_name(), TRITONSERVER_TYPE_INT16, dtype,
                            byte_size));
                        serialized_data->emplace_back();
                        auto& serialized = serialized_data->back();
                        serialized.reserve(
                            io.contents().int_contents_size() * elem_byte_size);
                        for (const auto& element : io.contents().int_contents()) {
                            // Assuming the system is little-endian, picking the
                            // least 2 significant bytes of 32-bit integer as a
                            // int16 element
                            serialized.append(
                                reinterpret_cast<const char*>(&element), elem_byte_size);
                        }
                        base = serialized.c_str();
                        byte_size = serialized.size();
                    } else {
                        RETURN_IF_ERR(InferKytensorToInputHelper(
                            io.name(), request.model_name(), TRITONSERVER_TYPE_INT32, dtype,
                            byte_size));
                        base = (const void*)io.contents().int_contents().data();
                        byte_size = io.contents().int_contents_size() * elem_byte_size;
                    }
                }

                if (io.contents().int64_contents_size() != 0) {
                    RETURN_IF_ERR(InferKytensorToInputHelper(
                        io.name(), request.model_name(), TRITONSERVER_TYPE_INT64, dtype,
                        byte_size));
                    base = (const void*)io.contents().int64_contents().data();
                    byte_size = io.contents().int64_contents_size() * elem_byte_size;
                }

                if (io.contents().uint_contents_size() != 0) {
                    if (dtype == TRITONSERVER_TYPE_UINT8) {
                        RETURN_IF_ERR(InferKytensorToInputHelper(
                            io.name(), request.model_name(), TRITONSERVER_TYPE_UINT8, dtype,
                            byte_size));
                        serialized_data->emplace_back();
                        auto& serialized = serialized_data->back();
                        serialized.reserve(
                            io.contents().uint_contents_size() * elem_byte_size);
                        for (const auto& element : io.contents().uint_contents()) {
                            // Assuming the system is little-endian, picking the
                            // least significant byte of 32-bit unsigned integer as a
                            // uint8 element
                            serialized.append(
                                reinterpret_cast<const char*>(&element), elem_byte_size);
                        }
                        base = serialized.c_str();
                        byte_size = serialized.size();
                    } else if (dtype == TRITONSERVER_TYPE_UINT16) {
                        RETURN_IF_ERR(InferKytensorToInputHelper(
                            io.name(), request.model_name(), TRITONSERVER_TYPE_UINT16,
                            dtype, byte_size));
                        serialized_data->emplace_back();
                        auto& serialized = serialized_data->back();
                        serialized.reserve(
                            io.contents().uint_contents_size() * elem_byte_size);
                        for (const auto& element : io.contents().uint_contents()) {
                            // Assuming the system is little-endian, picking the
                            // least 2 significant bytes of 32-bit integer as a
                            // uint16 element
                            serialized.append(
                                reinterpret_cast<const char*>(&element), elem_byte_size);
                        }
                        base = serialized.c_str();
                        byte_size = serialized.size();
                    } else {
                        RETURN_IF_ERR(InferKytensorToInputHelper(
                            io.name(), request.model_name(), TRITONSERVER_TYPE_UINT32,
                            dtype, byte_size));
                        base = (const void*)io.contents().uint_contents().data();
                        byte_size = io.contents().uint_contents_size() * elem_byte_size;
                    }
                }

                if (io.contents().uint64_contents_size() != 0) {
                    RETURN_IF_ERR(InferKytensorToInputHelper(
                        io.name(), request.model_name(), TRITONSERVER_TYPE_UINT64, dtype,
                        byte_size));
                    base = (const void*)io.contents().uint64_contents().data();
                    byte_size = io.contents().uint64_contents_size() * elem_byte_size;
                }

                if (io.contents().fp32_contents_size() != 0) {
                    RETURN_IF_ERR(InferKytensorToInputHelper(
                        io.name(), request.model_name(), TRITONSERVER_TYPE_FP32, dtype,
                        byte_size));
                    base = (const void*)io.contents().fp32_contents().data();
                    byte_size = io.contents().fp32_contents_size() * elem_byte_size;
                }

                if (io.contents().fp64_contents_size() != 0) {
                    RETURN_IF_ERR(InferKytensorToInputHelper(
                        io.name(), request.model_name(), TRITONSERVER_TYPE_FP64, dtype,
                        byte_size));
                    base = (const void*)io.contents().fp64_contents().data();
                    byte_size = io.contents().fp64_contents_size() * elem_byte_size;
                }

                if (io.contents().bytes_contents_size() != 0) {
                    RETURN_IF_ERR(InferKytensorToInputHelper(
                        io.name(), request.model_name(), TRITONSERVER_TYPE_BYTES, dtype,
                        byte_size));

                    serialized_data->emplace_back();
                    auto& serialized = serialized_data->back();

                    // Serialize the output tensor strings. Each string is
                    // serialized as a 4-byte length followed by the string itself
                    // with no null-terminator.
                    for (const auto& element : io.contents().bytes_contents()) {
                        uint32_t len{(uint32_t)element.size()};
                        serialized.append(
                            reinterpret_cast<const char*>(&len), sizeof(uint32_t));
                        if (element.size() > 0) {
                            serialized.append(element.c_str(), len);
                        }
                    }
                    base = serialized.c_str();
                    byte_size = serialized.size();
                }
            } else if (request.raw_input_contents().size() > index) {
                // Try to read the raw contents if available
                const std::string& raw = request.raw_input_contents()[index++];
                base = raw.c_str();
                byte_size = raw.size();
            } else {
                return TRITONSERVER_ErrorNew(
                    TRITONSERVER_ERROR_INVALID_ARG,
                    std::string(
                        "unable to find data for input tensor '" + io.name() +
                        "' for model '" + request.model_name() + "' in request.")
                        .c_str());
            }
        }

        if (cuda_ipc_handle != nullptr) {
            RETURN_IF_ERR(TRITONSERVER_BufferAttributesSetCudaIpcHandle(
                buffer_attributes, reinterpret_cast<void*>(cuda_ipc_handle)));
        }

        RETURN_IF_ERR(TRITONSERVER_BufferAttributesSetMemoryType(
            buffer_attributes, memory_type));
        RETURN_IF_ERR(TRITONSERVER_BufferAttributesSetMemoryTypeId(
            buffer_attributes, memory_type_id));
        RETURN_IF_ERR(
            TRITONSERVER_BufferAttributesSetByteSize(buffer_attributes, byte_size));
        RETURN_IF_ERR(
            TRITONSERVER_InferenceRequestAppendInputDataWithBufferAttributes(
                inference_request, io.name().c_str(), base, buffer_attributes));
    }

    return nullptr;  // success
}

void
InferRequestComplete(
    TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp)
{
    LOGF_DBG("ModelInferHandler::InferRequestComplete\n");

    RequestReleasePayload* request_release_payload =
        static_cast<RequestReleasePayload*>(userp);

    if ((flags & TRITONSERVER_REQUEST_RELEASE_ALL) != 0) {
        delete request_release_payload;
    }
}

}}