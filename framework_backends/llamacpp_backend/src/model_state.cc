#include "model_state.h"
#include "utils.h"

namespace triton::backend::llamacpp 
{

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, std::string const& name, uint64_t const version, ModelState** state)
{
	TRITONSERVER_Message* config_message;
    RETURN_IF_ERROR(TRITONBACKEND_ModelConfig(triton_model, 1 /* config_version */, &config_message));

	char const* buffer;
    size_t byte_size;
    RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(config_message, &buffer, &byte_size));

    common::TritonJson::Value model_config;
    TRITONSERVER_Error* err = model_config.Parse(buffer, byte_size);
    RETURN_IF_ERROR(TRITONSERVER_MessageDelete(config_message));
    RETURN_IF_ERROR(err);

	try {
		*state = new ModelState(triton_model, name, version, std::move(model_config));
	}
	catch (const BackendModelException& ex) {
		RETURN_ERROR_IF_TRUE(
			ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
			std::string("unexpected nullptr in BackendModelException"));
		RETURN_IF_ERROR(ex.err_);
	}

	return nullptr;  // success
}

void ModelState::load_params()
{
    n_threads_ = GetParameter<int32_t>("n_threads");
    n_predict_ = GetParameter<int32_t>("n_predict");
	n_ctx_ = GetParameter<int32_t>("n_ctx");
	n_batch_ = GetParameter<int32_t>("n_batch");
	n_parallel_ = GetParameter<int32_t>("n_parallel");
	model_ = GetParameter<std::string>("model");
	cont_batching_ = GetParameter<bool>("cont_batching");
    metrics_period_ms_ = GetParameter<int32_t>("metrics_period_ms");
    try
    {
        auto lora = GetParameter<std::string>("lora");

        auto lora_list = utils::split(lora, ';');
        
        if(lora_list.empty()) {
            lora_path_ = std::vector<std::string>{};
        } else {
            for (auto const& path : lora_list) {
                lora_path_.push_back(path);
            }
        }

    }
    catch(const std::exception& e)
    {
        LOG_MESSAGE(
            TRITONSERVER_LOG_WARN,
            std::string("lora is not specified ").c_str());
    }

    lora_init_without_apply_ = GetParameter<bool>("lora_init_without_apply");
    model_decrypt_type_ = GetParameter<int32_t>("model_decrypt_type");
    worker_save_path_ = GetParameter<std::string>("worker_save_path");
    use_jinja_ = GetParameter<bool>("use_jinja");
    chat_template_file_ = GetParameter<std::string>("chat_template_file");
    llamacpp_verbose_ = GetParameter<bool>("llamacpp_verbose");
}

template <>
std::string ModelState::GetParameter<std::string>(std::string const& name)
{
    TritonJson::Value parameters;
    TRITONSERVER_Error* err = model_config_.MemberAsObject("parameters", &parameters);
    if (err != nullptr)
    {
        TRITONSERVER_ErrorDelete(err);
        throw std::runtime_error("Model config doesn't have a parameters section");
    }
    TritonJson::Value value;
    std::string str_value;
    err = parameters.MemberAsObject(name.c_str(), &value);
    if (err != nullptr)
    {
        TRITONSERVER_ErrorDelete(err);
        std::string errStr = "Cannot find parameter with name: " + name;
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            (std::string("GetParameter: ") + errStr).c_str());
        return "";
        //throw std::runtime_error(errStr);
    }
    value.MemberAsString("string_value", &str_value);
    return str_value;
}

template <>
int32_t ModelState::GetParameter<int32_t>(std::string const& name)
{
    return std::stoi(GetParameter<std::string>(name));
}

#if 0
template <>
std::vector<int32_t> ModelState::GetParameter<std::vector<int32_t>>(std::string const& name)
{
    auto deviceIdsStr = GetParameter<std::string>(name);
    // Parse as comma delimited string
    return utils::csvStrToVecInt(deviceIdsStr);
}
#endif

template <>
uint32_t ModelState::GetParameter<uint32_t>(std::string const& name)
{
    return (uint32_t) std::stoul(GetParameter<std::string>(name));
}

template <>
int64_t ModelState::GetParameter<int64_t>(std::string const& name)
{
    return std::stoll(GetParameter<std::string>(name));
}

template <>
uint64_t ModelState::GetParameter<uint64_t>(std::string const& name)
{
    return std::stoull(GetParameter<std::string>(name));
}

template <>
float ModelState::GetParameter<float>(std::string const& name)
{
    return std::stof(GetParameter<std::string>(name));
}

template <>
bool ModelState::GetParameter<bool>(std::string const& name)
{
    auto val = GetParameter<std::string>(name);
    if (val == "True" || val == "true" || val == "TRUE" || val == "1")
    {
        return true;
    }
    else
    {
        return false;
    }
}

#if 0
template <>
std::vector<std::vector<int32_t>> ModelState::GetParameter<std::vector<std::vector<int32_t>>>(std::string const& name)
{
    auto str = GetParameter<std::string>(name);
    // Parse as comma delimited string and {} as array bounders
    return utils::csvStrToVecVecInt(str);
}
#endif 

} //namespace triton::backend::llamacpp 