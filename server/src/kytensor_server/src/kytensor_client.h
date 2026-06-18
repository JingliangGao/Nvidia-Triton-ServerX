#pragma once

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <cstdint>

#include "kytensor_client_common.h"
#include "kytensor_service.pb.h"

namespace kytensor { namespace client {

enum class KYTENSOR_COMPRESSION_TYPE {
    NONE = 0,
    GZIP = 1,
    SNAPPY = 2,
};

class KyTensorImpl;

class KyTensorClient {
public:

using OnCompleteFunc = std::function<void(const EmptyPtr&)>;
using OnCompleteFn = std::function<void(InferResult*)>;

    ~KyTensorClient();

    static Error Create(std::unique_ptr<KyTensorClient>* client, bool verbose = false);

    bool Start(bool verbose);
    void Stop();
    void SyncSendMessage(const std::string& content);
    void AsyncSendMessage(OnCompleteFunc callback, const int id);

    Error IsServerLive(bool* live, const uint64_t timeout_ms = 0);
    Error IsServerReady(bool* ready, const uint64_t timeout_ms = 0);
    Error IsModelReady(bool* ready, const std::string& model_name,
                     const std::string& model_version = "", const uint64_t timeout_ms = 0);
    Error ServerMetadata(inference::ServerMetadataResponse* server_metadata,
                       const uint64_t timeout_ms = 0);
    Error ModelMetadata(inference::ModelMetadataResponse* model_metadata,
                      const std::string& model_name, const std::string& model_version = "",
                      const uint64_t timeout_ms = 0);
    Error ModelConfig(inference::ModelConfigResponse* model_config,
                    const std::string& model_name, const std::string& model_version = "",
                    const uint64_t timeout_ms = 0);
    Error ModelRepositoryIndex(inference::RepositoryIndexResponse* repository_index,
                           const uint64_t timeout_ms = 0);
    Error LoadModel(const std::string& model_name,
                  const std::string& config = std::string(),
                  const std::map<std::string, std::vector<char>>& files = {},
                  const uint64_t timeout_ms = 0);
    Error UnloadModel(const std::string& model_name, const uint64_t timeout_ms = 0);
    Error ModelInferenceStatistics(inference::ModelStatisticsResponse* infer_stat,
                                const std::string& model_name = "", const std::string& model_version = "",
                                const uint64_t timeout_ms = 0);

    Error Infer(InferResult** result, const InferOptions& options,
               const std::vector<InferInput*>& inputs,
               const std::vector<const InferRequestedOutput*>& outputs =
                   std::vector<const InferRequestedOutput*>(),
               KYTENSOR_COMPRESSION_TYPE compression_type = KYTENSOR_COMPRESSION_TYPE::NONE);

    Error AsyncInfer(OnCompleteFn callback, const InferOptions& options,
                    const std::vector<InferInput*>& inputs,
                    const std::vector<const InferRequestedOutput*>& outputs =
                        std::vector<const InferRequestedOutput*>(),
                    KYTENSOR_COMPRESSION_TYPE compression_type = KYTENSOR_COMPRESSION_TYPE::NONE);

    Error StartStream(OnCompleteFn callback, bool enable_stats = true,
                     uint32_t stream_timeout = 0,
                     KYTENSOR_COMPRESSION_TYPE compression_type = KYTENSOR_COMPRESSION_TYPE::NONE);

    Error StopStream();

    Error AsyncStreamInfer(const InferOptions& options, const std::vector<InferInput*>& inputs,
                         const std::vector<const InferRequestedOutput*>& outputs =
                             std::vector<const InferRequestedOutput*>());

private:
    friend class KyTensorImpl;
    KyTensorClient(std::unique_ptr<KyTensorImpl> impl);

    std::unique_ptr<KyTensorImpl> pImpl;
};

}}