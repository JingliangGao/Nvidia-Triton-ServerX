#include "llamacpp_metrics.h"
#include "triton/backend/backend_common.h"
#include "iomanip"
#include "sstream"
#include <vector>

using namespace::triton::common; //TritonJson

namespace triton::backend::llamacpp 
{

const std::vector<std::string> LlamaCppMetrics::request_keys_{
    "Requests Processing", "Requests Deferred"};
const std::vector<std::string> LlamaCppMetrics::request_labels_{"processing", "deferred"};

const std::vector<std::string> LlamaCppMetrics::runtime_memory_keys_{
    "Runtime CPU Memory Usage", "Runtime GPU Memory Usage", "Runtime Pinned Memory Usage"};
const std::vector<std::string> LlamaCppMetrics::runtime_memory_labels_{"cpu", "gpu", "pinned"};

const std::vector<std::string> LlamaCppMetrics::kv_cache_keys_{
    "KV Cache Usage Ratio", "KV Cache Tokens"};
const std::vector<std::string> LlamaCppMetrics::kv_cache_labels_{"kv_cache_ratio", "kv_cache_tokens"};

const std::vector<std::string> LlamaCppMetrics::llamacpp_specific_keys_{
    "Prompt Tokens Seconds", "Prompt Seconds Total", "Tokens Predicted Total", 
    "Tokens Predicted Seconds Total", "Prompt Tokens Seconds", "Predicted Tokens Seconds"};
const std::vector<std::string> LlamaCppMetrics::llamacpp_specific_labels_{
    "prompt_tokens_seconds", "prompt_seconds_total", "tokens_predicted_total",
    "tokens_predicted_seconds_total", "prompt_tokens_seconds", "predicted_tokens_seconds"};

const std::vector<std::string> LlamaCppMetrics::general_metric_keys_{"Timestamp", "Iteration Counter"};
const std::vector<std::string> LlamaCppMetrics::general_metric_labels_{"timestamp", "iteration_counter"};

uint64_t ConvertTimestampToSeconds(std::string const& ts)
{
    std::tm tm = {};
    std::stringstream ss(ts);
    ss >> std::get_time(&tm, "%m-%d-%Y %H:%M:%S");
    auto timestamp = std::chrono::system_clock::from_time_t(std::mktime(&tm));
    auto epoch = std::chrono::time_point_cast<std::chrono::seconds>(timestamp).time_since_epoch();
    uint64_t time_in_seconds = std::chrono::duration_cast<std::chrono::seconds>(epoch).count();
    return time_in_seconds;
}

TritonMetricGroup::TritonMetricGroup(std::string const& metric_family_label,
    std::string const& metric_family_description, std::string const& category_label,
    std::vector<std::string> const& json_keys, std::vector<std::string> const& sub_labels)
    : metric_family_label_(metric_family_label)
    , metric_family_description_(metric_family_description)
    , category_label_(category_label)
    , json_keys_(json_keys)
    , sub_labels_(sub_labels)
{
}

TRITONSERVER_Error* TritonMetricGroup::CreateGroup(std::string const& model_name, const uint64_t version)
{
    TRITONSERVER_MetricFamily* metric_family = nullptr;
    RETURN_IF_ERROR(TRITONSERVER_MetricFamilyNew(&metric_family, TRITONSERVER_METRIC_KIND_GAUGE,
        metric_family_label_.c_str(), metric_family_description_.c_str()));
    metric_family_.reset(metric_family);

    std::vector<TRITONSERVER_Parameter const*> labels;
    std::unique_ptr<TRITONSERVER_Parameter, ParameterDeleter> model_label(
        TRITONSERVER_ParameterNew("model", TRITONSERVER_PARAMETER_STRING, model_name.c_str()));
    std::unique_ptr<TRITONSERVER_Parameter, ParameterDeleter> model_version(
        TRITONSERVER_ParameterNew("version", TRITONSERVER_PARAMETER_STRING, std::to_string(version).c_str()));
    labels.emplace_back(model_label.get());
    labels.emplace_back(model_version.get());

    for (size_t i = 0; i < sub_labels_.size(); ++i)
    {
        TRITONSERVER_Metric* metric;
        std::unique_ptr<TRITONSERVER_Parameter, ParameterDeleter> sub_label(
            TRITONSERVER_ParameterNew(category_label_.c_str(), TRITONSERVER_PARAMETER_STRING, sub_labels_[i].c_str()));
        labels.emplace_back(sub_label.get());
        RETURN_IF_ERROR(TRITONSERVER_MetricNew(&metric, metric_family_.get(), labels.data(), labels.size()));
        std::unique_ptr<TRITONSERVER_Metric, MetricDeleter> unique_metric(metric);
        metrics_.push_back(std::move(unique_metric));
        labels.pop_back();
    }

    return nullptr; // success
}

TRITONSERVER_Error* TritonMetricGroup::UpdateGroup(std::vector<uint64_t>& values)
{
    for (size_t i = 0; i < values.size(); ++i)
    {
        RETURN_IF_ERROR(TRITONSERVER_MetricSet(metrics_[i].get(), values[i]));
    }
    return nullptr; // success
}

std::vector<std::string> const& TritonMetricGroup::JsonKeys() const
{
    return json_keys_;
}

TRITONSERVER_Error* LlamaCppMetrics::InitLlamaCppMetrics(
    std::string const& model_name, const uint64_t version)
{
    /* REQUEST METRIC GROUP */
    request_metric_family_ = std::make_unique<TritonMetricGroup>(
        "llamacpp_request_metrics", "llamacpp request metrics", "request_type", request_keys_, request_labels_);

    RETURN_IF_ERROR(request_metric_family_->CreateGroup(model_name, version));
    metric_groups_.push_back(std::move(request_metric_family_));

    /* RUNTIME MEMORY METRIC GROUP */
    // runtime_memory_metric_family_ = std::make_unique<TritonMetricGroup>("llamacpp_runtime_memory_metrics",
    //     "llamacpp runtime memory metrics", "memory_type", runtime_memory_keys_, runtime_memory_labels_);

    // RETURN_IF_ERROR(runtime_memory_metric_family_->CreateGroup(model_name, version));
    // metric_groups_.push_back(std::move(runtime_memory_metric_family_));

    /* KV CACHE METRIC GROUP */
    kv_cache_metric_family_ = std::make_unique<TritonMetricGroup>("llamacpp_kv_cache_block_metrics",
        "llamacpp KV cache block metrics", "kv_cache_block_type", kv_cache_keys_, kv_cache_labels_);

    RETURN_IF_ERROR(kv_cache_metric_family_->CreateGroup(model_name, version));
    metric_groups_.push_back(std::move(kv_cache_metric_family_));

    std::string model = "llamacpp";
    std::string model_metric_family_label = model + "_metrics";
    std::string model_metric_family_description = model + "-specific metrics";
    std::string model_metric_family_category = model + "_specific_metric";

    llamacpp_metric_family_ = std::make_unique<TritonMetricGroup>(model_metric_family_label,
    model_metric_family_description, model_metric_family_category, llamacpp_specific_keys_, llamacpp_specific_labels_);
  

    RETURN_IF_ERROR(llamacpp_metric_family_->CreateGroup(model_name, version));
    metric_groups_.push_back(std::move(llamacpp_metric_family_));

    /* GENERAL METRIC GROUP */
    // general_metric_family_ = std::make_unique<TritonMetricGroup>("llm_general_metrics",
    //     "General llm metrics", "general_type", general_metric_keys_, general_metric_labels_);

    // RETURN_IF_ERROR(general_metric_family_->CreateGroup(model_name, version));
    // metric_groups_.push_back(std::move(general_metric_family_));

    return nullptr; // success
}

TRITONSERVER_Error* LlamaCppMetrics::UpdateLlamaCppMetrics(std::string const& custom_metrics)
{
    triton::common::TritonJson::Value metrics;
    std::vector<std::string> members;
    metrics.Parse(custom_metrics);
    metrics.Members(&members);

    for (auto const& metric_group : metric_groups_)
    {
        std::vector<std::string> metric_group_keys = metric_group->JsonKeys();
        std::vector<uint64_t> metric_group_values;
        uint64_t value;
        double dvalue = 0.0;
        for (auto const& key : metric_group_keys)
        {
            triton::common::TritonJson::Value value_json;
            if (!metrics.Find(key.c_str(), &value_json))
            {
                std::string errStr = std::string("Failed to find " + key + " in metrics.");
                return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, errStr.c_str());
            }
            if (key == "Timestamp")
            {
                std::string timestamp;
                value_json.AsString(&timestamp);
                value = ConvertTimestampToSeconds(timestamp);
                metric_group_values.push_back(value);
            }
            else if(key == "Prompt Seconds Total" || key == "Tokens Predicted Seconds Total" ||
                    key == "Prompt Tokens Seconds" || key == "Predicted Tokens Seconds" ||
                    key == "KV Cache Usage Ratio")
            {
                value_json.AsDouble(&dvalue);
                metric_group_values.push_back(dvalue);
            }
            else
            {
                value_json.AsUInt(&value);
                metric_group_values.push_back(value);
            }

        }

        RETURN_IF_ERROR(metric_group->UpdateGroup(metric_group_values));
    }

    return nullptr;
}

} // namespace triton::backend::llamacpp