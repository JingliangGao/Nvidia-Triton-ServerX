#pragma once

#include "llamacpp_queue.h"
#include "llamacpp_response.h"
#include "llamacpp_worker.h"

namespace triton::backend::llamacpp
{

struct server_metrics {
    int64_t t_start = 0;

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

    void init() {
        t_start = ggml_time_us();
    }

    void on_prompt_eval(const LlamaCppWorker & worker) {
        n_prompt_tokens_processed_total += worker.n_prompt_tokens_processed;
        n_prompt_tokens_processed       += worker.n_prompt_tokens_processed;
        t_prompt_processing             += worker.t_prompt_processing;
        t_prompt_processing_total       += worker.t_prompt_processing;
    }

    void on_prediction(const LlamaCppWorker & worker) {
        n_tokens_predicted_total   += worker.n_decoded;
        n_tokens_predicted         += worker.n_decoded;
        t_tokens_generation        += worker.t_token_generation;
        t_tokens_generation_total  += worker.t_token_generation;
    }

    void on_decoded(const std::vector<LlamaCppWorker> & workers) {
        n_decode_total++;
        for (const auto & worker : workers) {
            if (worker.is_processing()) {
                n_busy_workers_total++;
            }
        }
    }

    void reset_bucket() {
        n_prompt_tokens_processed = 0;
        t_prompt_processing       = 0;
        n_tokens_predicted        = 0;
        t_tokens_generation       = 0;
    }
};

class LlamaCppEngine {
public:
    common_params params_base;

    // note: keep these alive - they determine the lifetime of the model, context, etc.
    common_init_result llama_init;
    common_init_result llama_init_dft;

    llama_model * model = nullptr;
    llama_context * ctx = nullptr;

    const llama_vocab * vocab = nullptr;

    llama_model * model_dft = nullptr;

    llama_context_params cparams_dft;

    llama_batch batch = {};

    bool clean_kv_cache = true;
    bool add_bos_token  = true;
    bool has_eos_token  = false;

    int32_t n_ctx; // total context for all clients / workers

    // workers / clients
    std::vector<LlamaCppWorker> workers;
    json default_generation_settings_for_props;

    LlamaCppQueue    queue_tasks;
    LlamaCppResponse queue_results;

    server_metrics metrics;

    // Necessary similarity of prompt for worker selection
    float worker_prompt_similarity = 0.0f;

    common_chat_templates_ptr chat_templates;

    // used to determine if the decode should be aborted
    bool decode_abort_flag_ = false;
    static bool decode_abort_callback(void* user_data)
    {
        LlamaCppEngine *engine = static_cast<LlamaCppEngine*>(user_data);
        if(engine->decode_abort_flag_) {
            LOG_INF("decode abort callback called\n");
            engine->decode_abort_flag_ = false; // reset the flag
            return true;
        }
        return false;
    }

    // method
    ~LlamaCppEngine();
    bool load_model(const common_params & params);
    void init();
    LlamaCppWorker * get_worker_by_id(int id);
    LlamaCppWorker * get_available_worker(const ServerTask & task);
    bool can_be_detokenized(const struct llama_context * ctx, const std::vector<llama_token> & tokens);
    bool launch_worker_with_task(LlamaCppWorker & worker, const ServerTask & task);
    void kv_cache_clear();
    bool process_token(CompletionTokenOutput & result, LlamaCppWorker & worker);
    void populate_token_probs(const LlamaCppWorker & worker, CompletionTokenOutput & result, bool post_sampling, bool special, int idx);
    void send_error(const ServerTask & task, const std::string & error,
        const enum error_type type = ERROR_TYPE_SERVER);
    void send_error(const LlamaCppWorker & worker, const std::string & error,
        const enum error_type type = ERROR_TYPE_SERVER);
    void send_error(const int id_task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER);
    void send_partial_response(LlamaCppWorker & worker, const CompletionTokenOutput & tkn);
    void send_final_response(const LlamaCppWorker & worker);
    void send_embedding(const LlamaCppWorker & worker, const llama_batch & batch);
    void send_rerank(const LlamaCppWorker & worker, const llama_batch & batch);
    void cancel_tasks(const std::unordered_set<int> & id_tasks);
    void abort_tasks(const std::unordered_set<int> & id_tasks);
    void receive_multi_results(
        const std::unordered_set<int> & id_tasks,
        const std::function<void(std::vector<ServerTaskResultPtr>&)> & result_handler,
        const std::function<void(json)> & error_handler,
        const std::function<bool()> & is_connection_closed);
    void receive_cmpl_results_stream(
        const std::unordered_set<int> & id_tasks,
        const std::function<bool(ServerTaskResultPtr&)> & result_handler,
        const std::function<void(json)> & error_handler,
        const std::function<bool()> & is_connection_closed);
    void process_single_task(const ServerTask task);
    void update_workers();
    json model_meta() const;
};

} // namespace triton::backend::llamacpp