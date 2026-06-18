#pragma once

#include <string>
#include <google/protobuf/message.h>
#include <atomic>
#include <memory>

#include "triton/core/tritonserver.h"
#include "kytensor_service.pb.h"
#include "health.pb.h"
#include "../../shared_memory_manager.h"
#include "infer_response.h"
#include "kytensor_utils.h"
#include "../../common.h"
#include <event2/event.h>
#include <event2/bufferevent.h>

namespace kytensor { namespace server {

typedef std::shared_ptr<google::protobuf::Message> MessagePtr;
typedef std::shared_ptr<inference::Empty> EmptyPtr;
typedef std::shared_ptr<inference::ServerLiveRequest> ServerLiveRequestPtr;
typedef std::shared_ptr<inference::ServerLiveResponse> ServerLiveResponsePtr;
typedef std::shared_ptr<inference::ServerReadyRequest> ServerReadyRequestPtr;
typedef std::shared_ptr<inference::ServerReadyResponse> ServerReadyResponsePtr;
typedef std::shared_ptr<::grpc::health::v1::HealthCheckRequest> HealthCheckRequestPtr;
typedef std::shared_ptr<::grpc::health::v1::HealthCheckResponse> HealthCheckResponsePtr;
typedef std::shared_ptr<inference::ModelReadyRequest> ModelReadyRequestPtr;
typedef std::shared_ptr<inference::ModelReadyResponse> ModelReadyResponsePtr;
typedef std::shared_ptr<inference::ServerMetadataRequest> ServerMetadataRequestPtr;
typedef std::shared_ptr<inference::ServerMetadataResponse> ServerMetadataResponsePtr;
typedef std::shared_ptr<inference::ModelMetadataRequest> ModelMetadataRequestPtr;
typedef std::shared_ptr<inference::ModelMetadataResponse> ModelMetadataResponsePtr;
typedef std::shared_ptr<inference::ModelConfigRequest> ModelConfigRequestPtr;
typedef std::shared_ptr<inference::ModelConfigResponse> ModelConfigResponsePtr;
typedef std::shared_ptr<inference::ModelStatisticsRequest> ModelStatisticsRequestPtr;
typedef std::shared_ptr<inference::ModelStatisticsResponse> ModelStatisticsResponsePtr;
typedef std::shared_ptr<inference::ModelConfigRequest> ModelConfigRequestPtr;
typedef std::shared_ptr<inference::ModelConfigResponse> ModelConfigResponsePtr;
typedef std::shared_ptr<inference::RepositoryIndexRequest> RepositoryIndexRequestPtr;
typedef std::shared_ptr<inference::RepositoryIndexResponse> RepositoryIndexResponsePtr;
typedef std::shared_ptr<inference::RepositoryModelLoadRequest> RepositoryModelLoadRequestPtr;
typedef std::shared_ptr<inference::RepositoryModelLoadResponse> RepositoryModelLoadResponsePtr;
typedef std::shared_ptr<inference::RepositoryModelUnloadRequest> RepositoryModelUnloadRequestPtr;
typedef std::shared_ptr<inference::RepositoryModelUnloadResponse> RepositoryModelUnloadResponsePtr;
typedef std::shared_ptr<inference::ModelInferRequest> ModelInferRequestPtr;
typedef std::shared_ptr<inference::ModelInferResponse> ModelInferResponsePtr;

constexpr int MAX_GRPC_MESSAGE_SIZE = INT32_MAX;

#define GOTO_IF_ERR(X, T)            \
  do {                               \
    TRITONSERVER_Error* err__ = (X); \
    if (err__ != nullptr) {          \
      goto T;                        \
    }                                \
  } while (false)

//Connection for per client
class Connection : public std::enable_shared_from_this<Connection> {
public:
    explicit Connection(struct bufferevent* bev) : bev_(bev) {}
    ~Connection() {
        if (bev_) {
            bufferevent_free(bev_);
        }
    }

    struct bufferevent* bev() const { return bev_; }
    bool is_closed() const { return closed_.load(); }
    void set_closed() { closed_.store(true); }

private:
    struct bufferevent* bev_;
    std::atomic<bool> closed_{false};
};

struct StateParameters {
  // Whether to generate an empty response when a FINAL flag is received with
  // no corresponding response. Only applicable to StreamInferHandlerState.
  bool enable_empty_final_response_ = false;
};

// Simple structure that carries the userp payload needed for
// request release callback.
struct RequestReleasePayload final {
	explicit RequestReleasePayload(
		const std::shared_ptr<TRITONSERVER_InferenceRequest>& inference_request)
		: inference_request_(inference_request){};

   private:
	std::shared_ptr<TRITONSERVER_InferenceRequest> inference_request_ = nullptr;
};

template <typename ResponseType>
TRITONSERVER_Error*
InferAllocatorPayload(
    const std::shared_ptr<TRITONSERVER_Server>& tritonserver,
    const std::shared_ptr<triton::server::SharedMemoryManager>& shm_manager,
    const inference::ModelInferRequest& request,
    std::list<std::string>&& serialized_data,
    std::shared_ptr<InferResponseQueue<ResponseType>> response_queue,
    AllocPayload<ResponseType>* alloc_payload)
{
    alloc_payload->response_queue_ = response_queue;
    alloc_payload->shm_map_.clear();
    alloc_payload->classification_map_.clear();
    alloc_payload->serialized_data_ = std::move(serialized_data);

    // If any of the outputs use shared memory, then we must calculate
    // the memory address for that output and store it in the allocator
    // payload so that it is available when the allocation callback is
    // invoked.
    for (const auto& io : request.outputs()) {
		std::string region_name;
		int64_t offset;
		size_t byte_size;
		bool has_shared_memory;
		RETURN_IF_ERR(ParseSharedMemoryParams<
						inference::ModelInferRequest::InferRequestedOutputTensor>(
			io, &has_shared_memory, &region_name, &offset, &byte_size));

		bool has_classification;
		uint32_t classification_count;
		RETURN_IF_ERR(ParseClassificationParams(
			io, &has_classification, &classification_count));

		if (has_shared_memory && has_classification) {
			return TRITONSERVER_ErrorNew(
				TRITONSERVER_ERROR_INVALID_ARG,
				"output can't set both 'shared_memory_region' and "
				"'classification'");
		}

		if (has_shared_memory) {
			void* base;
			TRITONSERVER_MemoryType memory_type;
			int64_t memory_type_id;
			RETURN_IF_ERR(shm_manager->GetMemoryInfo(
				region_name, offset, byte_size, &base, &memory_type,
				&memory_type_id));

			if (memory_type == TRITONSERVER_MEMORY_GPU) {
	#ifdef TRITON_ENABLE_GPU
			char* cuda_handle;
			RETURN_IF_ERR(shm_manager->GetCUDAHandle(
				region_name, reinterpret_cast<cudaIpcMemHandle_t**>(&cuda_handle)));
			alloc_payload->shm_map_.emplace(
				io.name(),
				ShmInfo(base, byte_size, memory_type, memory_type_id, cuda_handle));
	#endif
			} else {
			alloc_payload->shm_map_.emplace(
				io.name(), ShmInfo(
								base, byte_size, memory_type, memory_type_id,
								nullptr /* cuda_ipc_handle */));
			}
		} else if (has_classification) {
			alloc_payload->classification_map_.emplace(
				io.name(), classification_count);
		}
    }

	return nullptr;  // Success
}

template <typename ResponseType>
TRITONSERVER_Error*
InferResponseCompleteCommon(
    TRITONSERVER_Server* server, TRITONSERVER_InferenceResponse* iresponse,
    inference::ModelInferResponse& response,
    const AllocPayload<ResponseType>& alloc_payload)
{
  RETURN_IF_ERR(TRITONSERVER_InferenceResponseError(iresponse));

  const char *model_name, *id;
  int64_t model_version;
  RETURN_IF_ERR(TRITONSERVER_InferenceResponseModel(
      iresponse, &model_name, &model_version));
  RETURN_IF_ERR(TRITONSERVER_InferenceResponseId(iresponse, &id));

  response.set_id(id);
  response.set_model_name(model_name);
  response.set_model_version(std::to_string(model_version));

  // Propagate response parameters.
  uint32_t parameter_count;
  RETURN_IF_ERR(TRITONSERVER_InferenceResponseParameterCount(
      iresponse, &parameter_count));
  for (uint32_t pidx = 0; pidx < parameter_count; ++pidx) {
    const char* name;
    TRITONSERVER_ParameterType type;
    const void* vvalue;
    RETURN_IF_ERR(TRITONSERVER_InferenceResponseParameter(
        iresponse, pidx, &name, &type, &vvalue));
    inference::InferParameter& param = (*response.mutable_parameters())[name];
    switch (type) {
      case TRITONSERVER_PARAMETER_BOOL:
        param.set_bool_param(*(reinterpret_cast<const bool*>(vvalue)));
        break;
      case TRITONSERVER_PARAMETER_INT:
        param.set_int64_param(*(reinterpret_cast<const int64_t*>(vvalue)));
        break;
      case TRITONSERVER_PARAMETER_STRING:
        param.set_string_param(reinterpret_cast<const char*>(vvalue));
        break;
      case TRITONSERVER_PARAMETER_DOUBLE:
        param.set_double_param(*(reinterpret_cast<const double*>(vvalue)));
        break;
      case TRITONSERVER_PARAMETER_BYTES:
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_UNSUPPORTED,
            "Response parameter of type 'TRITONSERVER_PARAMETER_BYTES' is not "
            "currently supported");
        break;
    }
  }

  // Go through each response output and transfer information to the
  // corresponding GRPC response output.
  uint32_t output_count;
  RETURN_IF_ERR(
      TRITONSERVER_InferenceResponseOutputCount(iresponse, &output_count));
  if (output_count != (uint32_t)response.outputs_size()) {
    //LOGF_INF("output_count:%d, response: %d\n", output_count, (uint32_t)response.outputs_size());
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, "response output count mismatch");
  }

  for (uint32_t output_idx = 0; output_idx < output_count; ++output_idx) {
    const char* cname;
    TRITONSERVER_DataType datatype;
    const int64_t* shape;
    uint64_t dim_count;
    const void* base;
    size_t byte_size;
    TRITONSERVER_MemoryType memory_type;
    int64_t memory_type_id;
    void* userp;

    RETURN_IF_ERR(TRITONSERVER_InferenceResponseOutput(
        iresponse, output_idx, &cname, &datatype, &shape, &dim_count, &base,
        &byte_size, &memory_type, &memory_type_id, &userp));

    const std::string name(cname);

    // There are usually very few outputs so fastest just to look for
    // the one we want... could create a map for cases where there are
    // a large number of outputs. Or rely on order to be same...
    inference::ModelInferResponse::InferOutputTensor* output = nullptr;
    for (auto& io : *(response.mutable_outputs())) {
      if (io.name() == name) {
        output = &io;
        break;
      }
    }

    if (output == nullptr) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          "unable to find expected response output");
    }

    // If this output was requested as classification then remove the
    // raw output from the response and instead return classification
    // results as a string tensor
    const auto itr = alloc_payload.classification_map_.find(name);
    if (itr == alloc_payload.classification_map_.end()) {
      // Not classification...
      output->set_datatype(TRITONSERVER_DataTypeString(datatype));
      for (size_t idx = 0; idx < dim_count; idx++) {
        output->add_shape(shape[idx]);
      }
    } else {
      // Classification
      const uint32_t classification_count = itr->second;

      // For classification need to determine the batch size, if any,
      // because need to use that to break up the response for each
      // batch entry.
      uint32_t batch_size = 0;

      uint32_t batch_flags;
      RETURN_IF_ERR(TRITONSERVER_ServerModelBatchProperties(
          server, model_name, model_version, &batch_flags,
          nullptr /* voidp */));
      if ((dim_count > 0) &&
          ((batch_flags & TRITONSERVER_BATCH_FIRST_DIM) != 0)) {
        batch_size = shape[0];
      }

      // Determine the batch1 byte size of the tensor... needed when
      // the response tensor batch-size > 1 so that we know how to
      // stride though the tensor data.
      size_t batch1_element_count = 1;
      for (size_t idx = ((batch_size == 0) ? 0 : 1); idx < dim_count; idx++) {
        batch1_element_count *= shape[idx];
      }

      const size_t batch1_byte_size =
          batch1_element_count * TRITONSERVER_DataTypeByteSize(datatype);

      // Create the classification contents
      std::string serialized;

      size_t class_offset = 0;
      for (uint32_t bs = 0; bs < std::max((uint32_t)1, batch_size); ++bs) {
        std::vector<std::string> class_strs;
        // RETURN_IF_ERR(TopkClassifications(
        //     iresponse, output_idx,
        //     reinterpret_cast<const char*>(base) + class_offset,
        //     ((class_offset + batch1_byte_size) > byte_size) ? 0
        //                                                     : batch1_byte_size,
        //     datatype, classification_count, &class_strs));

        // Serialize for binary representation...
        for (const auto& str : class_strs) {
          uint32_t len = str.size();
          serialized.append(reinterpret_cast<const char*>(&len), sizeof(len));
          if (len > 0) {
            serialized.append(str);
          }
        }

        class_offset += batch1_byte_size;
      }

      // Update the output with new datatype, shape and contents.
      output->set_datatype(
          TRITONSERVER_DataTypeString(TRITONSERVER_TYPE_BYTES));

      if (batch_size > 0) {
        output->add_shape(batch_size);
      }
      output->add_shape(
          std::min(classification_count, (uint32_t)batch1_element_count));

      (*response.mutable_raw_output_contents())[output_idx] =
          std::move(serialized);
    }
  }

  // Make sure response doesn't exceed GRPC limits.
  if (response.ByteSizeLong() > MAX_GRPC_MESSAGE_SIZE) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string(
            "Response has byte size " +
            std::to_string(response.ByteSizeLong()) +
            " which exceeds gRPC's byte size limit " + std::to_string(INT_MAX) +
            ".")
            .c_str());
  }

  return nullptr;  // success
}

TRITONSERVER_Error* GetModelVersionFromString(
	const std::string& version_string, int64_t* version);

TRITONSERVER_Error*
OutputBufferAttributesHelper(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    const TensorShmMap& shm_map,
    TRITONSERVER_BufferAttributes* buffer_attributes);

TRITONSERVER_Error*
OutputBufferQueryHelper(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t* byte_size, const TensorShmMap& shm_map,
    TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id);

TRITONSERVER_Error*
ResponseAllocatorHelper(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, inference::ModelInferResponse* response,
    const TensorShmMap& shm_map, void** buffer, void** buffer_userp,
    TRITONSERVER_MemoryType* actual_memory_type, int64_t* actual_memory_type_id);

TRITONSERVER_Error*
SetInferenceRequestMetadata(
    TRITONSERVER_InferenceRequest* inference_request,
    const inference::ModelInferRequest& request, StateParameters& state_params);

TRITONSERVER_Error*
InferKytensorToInput(
    const std::shared_ptr<TRITONSERVER_Server>& tritonserver,
    const std::shared_ptr<triton::server::SharedMemoryManager>& shm_manager,
    const inference::ModelInferRequest& request,
    std::list<std::string>* serialized_data,
    TRITONSERVER_InferenceRequest* inference_request);

void
InferRequestComplete(
	TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp);

}}