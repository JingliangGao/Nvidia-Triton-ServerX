#pragma once

#include "llamacpp_utils.h"

#include <nlohmann/json.hpp>
#include "llamacpp/llama.h"
#include "llamacpp/common.h"

namespace triton::backend::llamacpp
{
using json = nlohmann::ordered_json;

// https://community.openai.com/t/openai-chat-list-of-error-codes-and-types/357791/11
enum error_type {
    ERROR_TYPE_INVALID_REQUEST,
    ERROR_TYPE_AUTHENTICATION,
    ERROR_TYPE_SERVER,
    ERROR_TYPE_NOT_FOUND,
    ERROR_TYPE_PERMISSION,
    ERROR_TYPE_UNAVAILABLE, // custom error
    ERROR_TYPE_NOT_SUPPORTED, // custom error
    ERROR_TYPE_DECODE_ABORT, // custom error
};

enum ServerTaskType {
    SERVER_TASK_TYPE_COMPLETION,
    SERVER_TASK_TYPE_EMBEDDING,
    SERVER_TASK_TYPE_RERANK,
    SERVER_TASK_TYPE_INFILL,
    SERVER_TASK_TYPE_CANCEL,
    SERVER_TASK_TYPE_ABORT,
    SERVER_TASK_TYPE_NEXT_RESPONSE,
    SERVER_TASK_TYPE_METRICS,
    SERVER_TASK_TYPE_WORKER_SAVE,
    SERVER_TASK_TYPE_WORKER_RESTORE,
    SERVER_TASK_TYPE_WORKER_ERASE,
    SERVER_TASK_TYPE_SET_LORA,
};

enum StopType {
    STOP_TYPE_NONE,
    STOP_TYPE_EOS,
    STOP_TYPE_WORD,
    STOP_TYPE_LIMIT,
};

enum WorkerState {
    WORKER_STATE_IDLE,
    WORKER_STATE_STARTED, // TODO: this state is only used for setting up the initial prompt processing; maybe merge it with launch_worker_with_task in the future
    WORKER_STATE_PROCESSING_PROMPT,
    WORKER_STATE_DONE_PROMPT,
    WORKER_STATE_GENERATING,
};

enum OAICompatType {
    OAICOMPAT_TYPE_NONE,
    OAICOMPAT_TYPE_CHAT,
    OAICOMPAT_TYPE_COMPLETION,
    OAICOMPAT_TYPE_EMBEDDING,
};

class CompletionTokenOutput {

public:
    llama_token tok;
    float prob;
    std::string text_to_send;
    struct prob_info {
        llama_token tok;
        std::string txt;
        float prob;
    };
    std::vector<prob_info> probs;

    json to_json(bool post_sampling_probs) const {
        json probs_for_token = json::array();
        for (const auto & p : probs) {
            std::string txt(p.txt);
            txt.resize(validate_utf8(txt));
            probs_for_token.push_back(json {
                {"id",      p.tok},
                {"token",   txt},
                {"bytes",   str_to_bytes(p.txt)},
                {
                    post_sampling_probs ? "prob" : "logprob",
                    post_sampling_probs ? p.prob : logarithm(p.prob)
                },
            });
        }
        return probs_for_token;
    }

    static json probs_vector_to_json(const std::vector<CompletionTokenOutput> & probs, bool post_sampling_probs) {
        json out = json::array();
        for (const auto & p : probs) {
            std::string txt(p.text_to_send);
            txt.resize(validate_utf8(txt));
            out.push_back(json {
                {"id",           p.tok},
                {"token",        txt},
                {"bytes",        str_to_bytes(p.text_to_send)},
                {
                    post_sampling_probs ? "prob" : "logprob",
                    post_sampling_probs ? p.prob : logarithm(p.prob)
                },
                {
                    post_sampling_probs ? "top_probs" : "top_logprobs",
                    p.to_json(post_sampling_probs)
                },
            });
        }
        return out;
    }

    static float logarithm(float x) {
        // nlohmann::json converts -inf to null, so we need to prevent that
        return x == 0.0f ? std::numeric_limits<float>::lowest() : std::log(x);
    }

    static std::vector<unsigned char> str_to_bytes(const std::string & str) {
        std::vector<unsigned char> bytes;
        for (unsigned char c : str) {
            bytes.push_back(c);
        }
        return bytes;
    }
};

class ResultTimings {
public:
    int32_t prompt_n = -1;
    double prompt_ms;
    double prompt_per_token_ms;
    double prompt_per_second;

    int32_t predicted_n = -1;
    double predicted_ms;
    double predicted_per_token_ms;
    double predicted_per_second;

    json to_json() const {
        return {
            {"prompt_n",               prompt_n},
            {"prompt_ms",              prompt_ms},
            {"prompt_per_token_ms",    prompt_per_token_ms},
            {"prompt_per_second",      prompt_per_second},

            {"predicted_n",            predicted_n},
            {"predicted_ms",           predicted_ms},
            {"predicted_per_token_ms", predicted_per_token_ms},
            {"predicted_per_second",   predicted_per_second},
        };
    }
};

}
