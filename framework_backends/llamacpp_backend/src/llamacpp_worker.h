#pragma once

#include "llamacpp_common.h"

#include "llamacpp/common.h"
#include "llamacpp/llama.h"
#include "llamacpp/chat.h"
#include "llamacpp/sampling.h"
#include "llamacpp/speculative.h"

namespace triton::backend::llamacpp
{

class WorkerParams {
public:
    bool stream        = true;
    bool cache_prompt  = true; // remember the prompt to avoid reprocessing all prompt
    bool return_tokens = false;

    int32_t n_keep    =  0; // number of tokens to keep from initial prompt
    int32_t n_discard =  0; // number of tokens after n_keep that may be discarded when shifting context, 0 defaults to half
    int32_t n_predict = -1; // new tokens to predict
    int32_t n_indent  =  0; // mininum line indentation for the generated text in number of whitespace characters

    int64_t t_max_prompt_ms  = -1; // TODO: implement
    int64_t t_max_predict_ms = -1; // if positive, limit the generation phase to this time limit

    std::vector<common_adapter_lora_info> lora;

    std::vector<std::string> antiprompt;
    std::vector<std::string> response_fields;
    bool timings_per_token = false;
    bool post_sampling_probs = false;
    bool ignore_eos = false;

    struct common_params_sampling sampling;
    struct common_params_speculative speculative;

    // OAI-compat fields
    bool                  verbose                   = false;
    OAICompatType         oaicompat                 = OAICOMPAT_TYPE_NONE;
    std::string           oaicompat_model;
    std::string           oaicompat_cmpl_id;
    common_chat_format    oaicompat_chat_format     = COMMON_CHAT_FORMAT_CONTENT_ONLY;

    json to_json() const;
};

// worker is a client
class LlamaCppWorker {

public:
    int id;
    int id_task = -1;

    // only used for completion/embedding/infill/rerank
    ServerTaskType task_type = SERVER_TASK_TYPE_COMPLETION;

    llama_batch batch_spec = {};

    llama_context * ctx = nullptr;
    llama_context * ctx_dft = nullptr;

    common_speculative * spec = nullptr;

    std::vector<common_adapter_lora_info> lora;

    // the index relative to completion multi-task request
    size_t index = 0;

    class WorkerParams params;

    WorkerState state = WORKER_STATE_IDLE;

    // used to determine the worker that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx       = 0;  // context size per worker
    int32_t n_past      = 0;
    int32_t n_decoded   = 0;
    int32_t n_remaining = -1;
    int32_t i_batch     = -1;
    int32_t n_predict   = -1; // TODO: disambiguate from params.n_predict

    // n_prompt_tokens may not be equal to prompt_tokens.size(), because prompt maybe truncated
    int32_t n_prompt_tokens           = 0;
    int32_t n_prompt_tokens_processed = 0;

    // input prompt tokens
    llama_tokens prompt_tokens;

    size_t last_nl_pos = 0;

    std::string  generated_text;
    llama_tokens generated_tokens;

    llama_tokens cache_tokens;

    std::vector<CompletionTokenOutput> generated_token_probs;

    bool has_next_token = true;
    bool has_new_line   = false;
    bool truncated      = false;
    StopType stop;

    std::string stopping_word;

    // sampling
    json json_schema;

    struct common_sampler * smpl = nullptr;

    llama_token sampled;

    common_chat_format chat_format = COMMON_CHAT_FORMAT_CONTENT_ONLY;

    // stats
    size_t n_sent_text        = 0; // number of sent text character

    int64_t t_start_process_prompt;
    int64_t t_start_generation;

    double t_prompt_processing; // ms
    double t_token_generation;  // ms

    std::function<void(int)> callback_on_release;

    // method

    void reset();
    bool is_non_causal() const {
        return task_type == SERVER_TASK_TYPE_EMBEDDING || task_type == SERVER_TASK_TYPE_RERANK;
    }

    bool can_batch_with(LlamaCppWorker & other_worker) const {
        return is_non_causal() == other_worker.is_non_causal()
            && are_lora_equal(lora, other_worker.lora);
    }

    bool has_budget(const common_params & global_params);
    bool is_processing() const;
    bool can_speculate() const;
    void add_token(const CompletionTokenOutput & token);
    void release();
    ResultTimings get_timings() const;
    size_t find_stopping_strings(const std::string & text, const size_t last_token_size, bool is_full_stop);
    void print_timings() const;
    json to_json() const;

private:

};

} // namespace triton::backend::llamacpp