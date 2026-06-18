#pragma once

#include "triton/backend/backend_common.h"
#include "triton/backend/backend_model.h"
#include "triton/core/tritonbackend.h"

#include "llamacpp_engine.h"

using namespace ::triton::common; // TritonJson

namespace triton::backend::llamacpp
{

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model. ModelState is derived from BackendModel class
// provided in the backend utilities that provides many common
// functions.
//
class ModelState {
public:
	static TRITONSERVER_Error* Create(
		TRITONBACKEND_Model* triton_model, std::string const& name, uint64_t const version, ModelState** state);
	virtual ~ModelState() = default;

	template <typename T>
    T GetParameter(std::string const& name)
    {
        assert(false);
        auto dummy = T();
        return dummy;
    }

	common::TritonJson::Value& GetModelConfig();
    std::string const& GetModelName() const {
        return model_name_;
    };
    uint64_t GetModelVersion() const {
        return model_version_;
    };


private:
	std::string const model_name_;
    uint64_t model_version_;
    common::TritonJson::Value model_config_;

	void load_params();

public:
	ModelState(
        TRITONBACKEND_Model* triton_model, std::string const& name, uint64_t version, TritonJson::Value&& model_config)
        : model_name_(name)
        , model_version_(version)
        , model_config_(std::move(model_config))
    {
		load_params();
    }

	int32_t n_threads_ = -1;
    int32_t n_predict_ = 512;
	int32_t n_ctx_ = 2048;
	int32_t n_batch_ = 512;
	int32_t n_parallel_ = 4;
	int32_t n_gpu_layers = 256;
	std::string model_;
	bool cont_batching_ = true;
    int32_t metrics_period_ms_ = 100;
    std::vector<std::string> lora_path_;
    bool lora_init_without_apply_ = false;
    int32_t model_decrypt_type_ = 0;
    std::string worker_save_path_ = "";
    bool use_jinja_ = false;
    std::string chat_template_file_ = "";
    bool llamacpp_verbose_ = false;
};

template <>
std::string ModelState::GetParameter<std::string>(std::string const& name);

template <>
int32_t ModelState::GetParameter<int32_t>(std::string const& name);

template <>
uint32_t ModelState::GetParameter<uint32_t>(std::string const& name);

template <>
int64_t ModelState::GetParameter<int64_t>(std::string const& name);

template <>
uint64_t ModelState::GetParameter<uint64_t>(std::string const& name);

template <>
float ModelState::GetParameter<float>(std::string const& name);

template <>
bool ModelState::GetParameter<bool>(std::string const& name);

//template <>
//std::vector<int32_t> ModelState::GetParameter<std::vector<int32_t>>(std::string const& name);

//template <>
//std::vector<std::vector<int32_t>> ModelState::GetParameter<std::vector<std::vector<int32_t>>>(std::string const& name);

} // namespace triton::backend::llamacpp