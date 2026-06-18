#include "kytensor_utils.h"

namespace kytensor { namespace server {

TRITONSERVER_Error*
ParseClassificationParams(
    const inference::ModelInferRequest::InferRequestedOutputTensor& output,
    bool* has_classification, uint32_t* classification_count)
{
    *has_classification = false;

    const auto& class_it = output.parameters().find("classification");
    if (class_it != output.parameters().end()) {
    *has_classification = true;

    const auto& param = class_it->second;
    if (param.parameter_choice_case() !=
        inference::InferParameter::ParameterChoiceCase::kInt64Param) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "invalid value type for 'classification' parameter, expected "
            "int64_param");
    }

    const int64_t cnt = param.int64_param();
    if (cnt <= 0) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "invalid value for 'classification' parameter, expected >= 0");
    }

    *classification_count = cnt;
    }

    return nullptr;  // success
}

}}