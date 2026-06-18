#pragma once
#include "llamacpp_worker.h"

#include <unordered_set>

namespace triton::backend::llamacpp
{

class ServerTask {

public:
    int id    = -1; // to be filled by server_queue
    int index = -1; // used when there are multiple prompts (batch request)

    ServerTaskType type;

    // used by SERVER_TASK_TYPE_CANCEL
    int id_target = -1;

    // used by SERVER_TASK_TYPE_INFERENCE
    WorkerParams  params;
    llama_tokens prompt_tokens;
    int id_selected_worker = -1;

    // used by SERVER_TASK_TYPE_WORKER_SAVE, SERVER_TASK_TYPE_WORKER_RESTORE, SERVER_TASK_TYPE_WORKER_ERASE
    struct worker_action {
        int worker_id;
        std::string filename;
        std::string filepath;
    };
    worker_action worker_action;

    // used by SERVER_TASK_TYPE_METRICS
    bool metrics_reset_bucket = false;

    // used by SERVER_TASK_TYPE_SET_LORA
    std::vector<common_adapter_lora_info> set_lora;

    ServerTask(ServerTaskType type) : type(type) {}

    static WorkerParams params_from_json_cmpl(
        const llama_context * ctx,
        const common_params & params_base,
        const json & data);

    // utility function
    static std::unordered_set<int> get_list_id(const std::vector<ServerTask> & tasks);
};

class ServerTaskResult {
public:
    int id           = -1;
    int id_worker      = -1;
    virtual bool is_error() {
        // only used by server_task_result_error
        return false;
    }
    virtual bool is_stop() {
        // only used by server_task_result_cmpl_*
        return false;
    }
    virtual int get_index() {
        return -1;
    }
    virtual json to_json() = 0;
    virtual ~ServerTaskResult() = default;
};

// using shared_ptr for polymorphism of server_task_result
using ServerTaskResultPtr = std::unique_ptr<ServerTaskResult>;



class ServerTaskResultCmplFinal : public ServerTaskResult {
public:
    int index = 0;

    std::string content;
    llama_tokens tokens;

    bool stream;
    ResultTimings timings;
    std::string prompt;

    bool truncated;
    int32_t n_decoded;
    int32_t n_prompt_tokens;
    int32_t n_tokens_cached;
    bool has_new_line;
    std::string stopping_word;
    StopType stop = STOP_TYPE_NONE;

    bool post_sampling_probs;
    std::vector<CompletionTokenOutput> probs_output;
    std::vector<std::string>  response_fields;

    WorkerParams generation_params;

    // OAI-compat fields
    bool                  verbose                  = false;
    OAICompatType        oaicompat                = OAICOMPAT_TYPE_NONE;
    std::string           oaicompat_model;
    std::string           oaicompat_cmpl_id;
    common_chat_format    oaicompat_chat_format    = COMMON_CHAT_FORMAT_CONTENT_ONLY;

    virtual int get_index() override {
        return index;
    }

    virtual bool is_stop() override {
        return true; // in stream mode, final responses are considered stop
    }

    virtual json to_json() override {
        switch (oaicompat) {
            case OAICOMPAT_TYPE_NONE:
                return to_json_non_oaicompat();
            case OAICOMPAT_TYPE_COMPLETION:
                return to_json_oaicompat();
            case OAICOMPAT_TYPE_CHAT:
                return stream ? to_json_oaicompat_chat_stream() : to_json_oaicompat_chat();
            default:
                GGML_ASSERT(false && "Invalid oaicompat_type");
        }
    }

    json to_json_non_oaicompat();
    json to_json_oaicompat();
    json to_json_oaicompat_chat();
    json to_json_oaicompat_chat_stream();
};

struct ServerTaskResultCmplPartial : public ServerTaskResult {
    int index = 0;

    std::string  content;
    llama_tokens tokens;

    int32_t n_decoded;
    int32_t n_prompt_tokens;

    bool post_sampling_probs;
    CompletionTokenOutput prob_output;
    ResultTimings timings;

    // OAI-compat fields
    bool           verbose   = false;
    OAICompatType oaicompat = OAICOMPAT_TYPE_NONE;
    std::string    oaicompat_model;
    std::string    oaicompat_cmpl_id;

    virtual int get_index() override {
        return index;
    }

    virtual bool is_stop() override {
        return false; // in stream mode, partial responses are not considered stop
    }

    virtual json to_json() override {
        switch (oaicompat) {
            case OAICOMPAT_TYPE_NONE:
                return to_json_non_oaicompat();
            case OAICOMPAT_TYPE_COMPLETION:
                return to_json_oaicompat();
            case OAICOMPAT_TYPE_CHAT:
                return to_json_oaicompat_chat();
            default:
                GGML_ASSERT(false && "Invalid oaicompat_type");
        }
    }

    json to_json_non_oaicompat();
    json to_json_oaicompat();
    json to_json_oaicompat_chat();
};

class ServerTaskResultEmbd : public ServerTaskResult {
public:
    int index = 0;
    std::vector<std::vector<float>> embedding;

    int32_t n_tokens;

    // OAI-compat fields
    OAICompatType oaicompat = OAICOMPAT_TYPE_NONE;

    virtual int get_index() override {
        return index;
    }

    virtual json to_json() override {
        return oaicompat == OAICOMPAT_TYPE_EMBEDDING
            ? to_json_oaicompat()
            : to_json_non_oaicompat();
    }

    json to_json_non_oaicompat() {
        return json {
            {"index",     index},
            {"embedding", embedding},
        };
    }

    json to_json_oaicompat() {
        return json {
            {"index",            index},
            {"embedding",        embedding[0]},
            {"tokens_evaluated", n_tokens},
        };
    }
};

class ServerTaskResultRerank : public ServerTaskResult {
public:
    int index = 0;
    float score = -1e6;

    int32_t n_tokens;

    virtual int get_index() override {
        return index;
    }

    virtual json to_json() override {
        return json {
            {"index",            index},
            {"score",            score},
            {"tokens_evaluated", n_tokens},
        };
    }
};

static json format_error_response(const std::string & message, const enum error_type type);

class ServerTaskResultError : public ServerTaskResult {
public:
    int index = 0;
    error_type err_type = ERROR_TYPE_SERVER;
    std::string err_msg;

    virtual bool is_error() override;

    virtual json to_json() override;
};

class ServerTaskResultMetrics : public ServerTaskResult {
public:
    int n_idle_workers;
    int n_processing_workers;
    int n_tasks_deferred;
    int64_t t_start;

    int32_t kv_cache_tokens_count;
    int32_t kv_cache_used_cells;

    // TODO: somehow reuse server_metrics in the future, instead of duplicating the fields
    uint64_t n_prompt_tokens_processed_total = 0;
    uint64_t t_prompt_processing_total       = 0;
    uint64_t n_tokens_predicted_total        = 0;
    uint64_t t_tokens_generation_total       = 0;

    uint64_t n_prompt_tokens_processed = 0;
    uint64_t t_prompt_processing       = 0;

    uint64_t n_tokens_predicted  = 0;
    uint64_t t_tokens_generation = 0;

    uint64_t n_decode_total     = 0;
    uint64_t n_busy_workers_total = 0;

    // while we can also use std::vector<server_worker> this requires copying the worker object which can be quite messy
    // therefore, we use json to temporarily store the worker.to_json() result
    json workers_data = json::array();

    virtual json to_json() override {
        return json {
            { "idle",                            n_idle_workers },
            { "processing",                      n_processing_workers },
            { "deferred",                        n_tasks_deferred },
            { "t_start",                         t_start },

            { "n_prompt_tokens_processed_total", n_prompt_tokens_processed_total },
            { "t_tokens_generation_total",       t_tokens_generation_total },
            { "n_tokens_predicted_total",        n_tokens_predicted_total },
            { "t_prompt_processing_total",       t_prompt_processing_total },

            { "n_prompt_tokens_processed",       n_prompt_tokens_processed },
            { "t_prompt_processing",             t_prompt_processing },
            { "n_tokens_predicted",              n_tokens_predicted },
            { "t_tokens_generation",             t_tokens_generation },

            { "n_decode_total",                  n_decode_total },
            { "n_busy_workers_total",              n_busy_workers_total },

            { "kv_cache_tokens_count",           kv_cache_tokens_count },
            { "kv_cache_used_cells",             kv_cache_used_cells },

            { "workers",                           workers_data },
        };
    }
};

class ServerTaskResultWorkerSaveLoad : public ServerTaskResult {
public:
    std::string filename;
    bool is_save; // true = save, false = load

    size_t n_tokens;
    size_t n_bytes;
    double t_ms;

    virtual json to_json() override {
        if (is_save) {
            return json {
                { "id_worker",   id_worker },
                { "filename",  filename },
                { "n_saved",   n_tokens },
                { "n_written", n_bytes },
                { "timings", {
                    { "save_ms", t_ms }
                }},
            };
        } else {
            return json {
                { "id_worker",    id_worker },
                { "filename",   filename },
                { "n_restored", n_tokens },
                { "n_read",     n_bytes },
                { "timings", {
                    { "restore_ms", t_ms }
                }},
            };
        }
    }
};

class ServerTaskResultWorkerErase : public ServerTaskResult {
public:
    size_t n_erased;

    virtual json to_json() override {
        return json {
            { "id_worker",  id_worker },
            { "n_erased", n_erased },
        };
    }
};

class ServerTaskResultApplyLora : public ServerTaskResult {
public:
    virtual json to_json() override {
        return json {{ "success", true }};
    }
};

}