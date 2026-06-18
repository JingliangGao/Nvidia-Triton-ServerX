#pragma once

#include "triton/core/tritonbackend.h"
#include "triton/core/tritonserver.h"
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace triton::backend::llamacpp
{

/// TritonMetricGroups are handled by the CustomMetricsReporter class
/// and encapsulate the creation/update functionality for a
/// group of TRT LLM statistics to be reported as custom Triton metrics.
/// The statistics (or custom metrics) handled by this class should
/// not be confused with Triton base metrics.
class TritonMetricGroup
{
public:
    TritonMetricGroup(std::string const& metric_family_label, std::string const& metric_family_description,
        std::string const& category_label, std::vector<std::string> const& json_keys,
        std::vector<std::string> const& labels);
    ~TritonMetricGroup(){};

    /// Create a new Triton metric family with corresponding metric
    /// pointers and parameters.
    ///
    /// \param model_name The name of the model to provide a metrics
    /// group for.
    /// \param version The version of the model to provide a metrics
    /// group for.
    /// \return a TRITONSERVER_Error indicating success or failure.
    TRITONSERVER_Error* CreateGroup(std::string const& model_name, const uint64_t version);

    /// Update the Triton metrics associated with this group using
    /// the parsed TRT LLM backend statistics values.
    ///
    /// \param values Values parsed from the TRT LLM backend
    /// statistics output, filtered by this group's JSON keys.
    /// \return a TRITONSERVER_Error indicating success or failure.
    TRITONSERVER_Error* UpdateGroup(std::vector<uint64_t>& values);

    /// Return a list of JSON keys that correspond to the TRT LLM
    /// statistics handled by this metric group.
    ///
    /// \return A const reference to vector of strings corresponding
    /// to the JSON keys associated with this group.
    std::vector<std::string> const& JsonKeys() const;

    /// Custom deleter for a unique TRITONSERVER_MetricFamily pointer
    struct MetricFamilyDeleter
    {
        void operator()(TRITONSERVER_MetricFamily* family)
        {
            if (family != nullptr)
            {
                TRITONSERVER_MetricFamilyDelete(family);
            }
        }
    };

    /// Custom deleter for a unique TRITONSERVER_Metric pointer
    struct MetricDeleter
    {
        void operator()(TRITONSERVER_Metric* metric)
        {
            if (metric != nullptr)
            {
                TRITONSERVER_MetricDelete(metric);
            }
        }
    };

    /// Custom deleter for a unique TRITONSERVER_Parameter pointer
    struct ParameterDeleter
    {
        void operator()(TRITONSERVER_Parameter* parameter)
        {
            if (parameter != nullptr)
            {
                TRITONSERVER_ParameterDelete(parameter);
            }
        }
    };

private:
    std::unique_ptr<TRITONSERVER_MetricFamily, MetricFamilyDeleter> metric_family_;
    std::vector<std::unique_ptr<TRITONSERVER_Metric, MetricDeleter>> metrics_;
    std::string metric_family_label_;
    std::string metric_family_description_;
    std::string category_label_;
    std::vector<std::string> json_keys_;
    std::vector<std::string> sub_labels_;
};

/// LlamaCppMetrics is an interface class meant to facilitate the
/// connection between llamacpp backend statistics and Triton custom metrics.
/// It functions by passing BatchManager statistics data from
/// the llamacpp backend to the multiple TritonMetricsGroup objects
/// it handles.
class LlamaCppMetrics
{
public:
    LlamaCppMetrics() {};
    ~LlamaCppMetrics() {};

    /// Initialize the various TritonMetricGroups handled by
    /// by this class using the static key/label members below.
    ///
    /// \param model The name of the model to provide metrics for.
    /// \param version The version of the model to provide metrics for.
    /// \return a TRITONSERVER_Error indicating success or failure.
    TRITONSERVER_Error* InitLlamaCppMetrics(std::string const& model, const uint64_t version);

    /// Updates the vector of TritonMetricGroup objects with a
    /// JSON-formatted statistics string.
    ///
    /// \param statistics A JSON-formatted string of llamacpp backend
    /// statistics.
    /// \return a TRITONSERVER_Error indicating success or failure.
    TRITONSERVER_Error* UpdateLlamaCppMetrics(std::string const& llamacpp_metrics);

    static const std::vector<std::string> request_keys_;
    static const std::vector<std::string> request_labels_;

    static const std::vector<std::string> runtime_memory_keys_;
    static const std::vector<std::string> runtime_memory_labels_;

    static const std::vector<std::string> kv_cache_keys_;
    static const std::vector<std::string> kv_cache_labels_;

    static const std::vector<std::string> llamacpp_specific_keys_;
    static const std::vector<std::string> llamacpp_specific_labels_;

    static const std::vector<std::string> general_metric_keys_;
    static const std::vector<std::string> general_metric_labels_;

private:
    std::vector<std::unique_ptr<TritonMetricGroup>> metric_groups_;
    std::unique_ptr<TritonMetricGroup> request_metric_family_;
    std::unique_ptr<TritonMetricGroup> runtime_memory_metric_family_;
    std::unique_ptr<TritonMetricGroup> kv_cache_metric_family_;
    std::unique_ptr<TritonMetricGroup> llamacpp_metric_family_;
    std::unique_ptr<TritonMetricGroup> general_metric_family_;
};


} // namespace triton::backend::llamacpp