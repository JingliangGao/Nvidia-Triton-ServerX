#include "llamacpp_engine.h"
#include "llamacpp_task.h"
#include "llamacpp/json-schema-to-grammar.h"
#include "llamacpp_utils.h"

#include <cinttypes>

namespace triton::backend::llamacpp
{

constexpr int LLAMA_POLLING_SECONDS = 1;

LlamaCppEngine::~LlamaCppEngine() {
    // Clear any sampling context
    for (LlamaCppWorker & worker : workers) {
        common_sampler_free(worker.smpl);
        worker.smpl = nullptr;

        llama_free(worker.ctx_dft);
        worker.ctx_dft = nullptr;

        common_speculative_free(worker.spec);
        worker.spec = nullptr;

        llama_batch_free(worker.batch_spec);
    }

    llama_batch_free(batch);
}

bool LlamaCppEngine::load_model(const common_params & params) {
    SRV_INF("loading model '%s'\n", params.model.c_str());

    params_base = params;

    llama_init = common_init_from_params(params_base);

    if (params_base.ctx_shift == false) {
        params_base.n_cache_reuse = 0;
        SRV_WRN("%s\n", "context shifting disabled, n_cache_reuse set to 0\n");
    }

    model = llama_init.model.get();
    ctx   = llama_init.context.get();

    if (model == nullptr) {
        SRV_ERR("failed to load model, '%s'\n", params_base.model.c_str());
        return false;
    }

    vocab = llama_model_get_vocab(model);

    n_ctx = llama_n_ctx(ctx);

    add_bos_token = llama_vocab_get_add_bos(vocab);
    has_eos_token = llama_vocab_eos(vocab) != LLAMA_TOKEN_NULL;

    if (!params_base.speculative.model.empty() || !params_base.speculative.hf_repo.empty()) {
        SRV_INF("loading draft model '%s'\n", params_base.speculative.model.c_str());

        auto params_dft = params_base;

        params_dft.devices      = params_base.speculative.devices;
        params_dft.hf_file      = params_base.speculative.hf_file;
        params_dft.hf_repo      = params_base.speculative.hf_repo;
        params_dft.model        = params_base.speculative.model;
        params_dft.model_url    = params_base.speculative.model_url;
        params_dft.n_ctx        = params_base.speculative.n_ctx == 0 ? params_base.n_ctx / params_base.n_parallel : params_base.speculative.n_ctx;
        params_dft.n_gpu_layers = params_base.speculative.n_gpu_layers;
        params_dft.n_parallel   = 1;

        // force F16 KV cache for the draft model for extra performance
        params_dft.cache_type_k = GGML_TYPE_F16;
        params_dft.cache_type_v = GGML_TYPE_F16;

        llama_init_dft = common_init_from_params(params_dft);

        model_dft = llama_init_dft.model.get();

        if (model_dft == nullptr) {
            SRV_ERR("failed to load draft model, '%s'\n", params_base.speculative.model.c_str());
            return false;
        }

        if (!common_speculative_are_compatible(ctx, llama_init_dft.context.get())) {
            SRV_ERR("the draft model '%s' is not compatible with the target model '%s'\n", params_base.speculative.model.c_str(), params_base.model.c_str());

            return false;
        }

        const int n_ctx_dft = llama_n_ctx(llama_init_dft.context.get());

        cparams_dft = common_context_params_to_llama(params_dft);
        cparams_dft.n_batch = n_ctx_dft;

        // the context is not needed - we will create one for each worker
        llama_init_dft.context.reset();
    }

    chat_templates = common_chat_templates_init(model, params_base.chat_template);
    try {
        common_chat_format_example(chat_templates.get(), params.use_jinja);
    } catch (const std::exception & e) {
        SRV_WRN("%s: Chat template parsing error: %s\n", __func__, e.what());
        SRV_WRN("%s: The chat template that comes with this model is not yet supported, falling back to chatml. This may cause the model to output suboptimal responses\n", __func__);
        chat_templates = common_chat_templates_init(model, "chatml");
    }

    //set abort callback
    llama_set_abort_callback(ctx, decode_abort_callback, this);

    return true;
}

void LlamaCppEngine::init() {
    const int32_t n_ctx_worker = n_ctx / params_base.n_parallel;

    SRV_INF("initializing workers, n_workers = %d\n", params_base.n_parallel);

    for (int i = 0; i < params_base.n_parallel; i++) {
        LlamaCppWorker worker;

        worker.id = i;
        worker.ctx = ctx;
        worker.n_ctx = n_ctx_worker;
        worker.n_predict = params_base.n_predict;

        if (model_dft) {
            worker.batch_spec = llama_batch_init(params_base.speculative.n_max + 1, 0, 1);

            worker.ctx_dft = llama_init_from_model(model_dft, cparams_dft);
            if (worker.ctx_dft == nullptr) {
                SRV_ERR("%s", "failed to create draft context\n");
                return;
            }

            worker.spec = common_speculative_init(worker.ctx_dft);
            if (worker.spec == nullptr) {
                SRV_ERR("%s", "failed to create speculator\n");
                return;
            }
        }

        SLT_INF(worker, "new worker n_ctx_worker = %d\n", worker.n_ctx);

        worker.params.sampling = params_base.sampling;

        worker.callback_on_release = [this](int) {
            queue_tasks.pop_deferred_task();
        };

        worker.reset();

        workers.push_back(worker);
    }

    default_generation_settings_for_props = workers[0].to_json();

    // the update_workers() logic will always submit a maximum of n_batch or n_parallel tokens
    // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not used)
    {
        const int32_t n_batch = llama_n_batch(ctx);

        // only a single seq_id per token is needed
        batch = llama_batch_init(std::max(n_batch, params_base.n_parallel), 0, 1);
    }

    metrics.init();
}

LlamaCppWorker * LlamaCppEngine::get_worker_by_id(int id) {
    for (LlamaCppWorker & worker : workers) {
        if (worker.id == id) {
            return &worker;
        }
    }

    return nullptr;
}

LlamaCppWorker * LlamaCppEngine::get_available_worker(const ServerTask & task) {
    LlamaCppWorker * ret = nullptr;

    // find the worker that has at least n% prompt similarity
    if (ret == nullptr && worker_prompt_similarity != 0.0f) {
        int max_lcs_len = 0;
        float max_similarity = 0;

        for (LlamaCppWorker & worker : workers) {
            // skip the worker if it is not available
            if (worker.is_processing()) {
                continue;
            }

            // skip the worker if it does not contains cached tokens
            if (worker.cache_tokens.empty()) {
                continue;
            }

            // length of the Longest Common Subsequence between the current worker's prompt and the input prompt
            int cur_lcs_len = common_lcs(worker.cache_tokens, task.prompt_tokens);

            // fraction of the common subsequence length compared to the current worker's prompt length
            float cur_similarity = static_cast<float>(cur_lcs_len) / static_cast<int>(worker.cache_tokens.size());

            //SLT_INF(worker, "selected worker by lcs similarity, lcs_len = %d, similarity = %f\n", cur_lcs_len, cur_similarity);

            // select the current worker if the criteria match
            if (cur_lcs_len > max_lcs_len && cur_similarity > max_similarity) {
                max_lcs_len = cur_lcs_len;
                max_similarity = cur_similarity;
                ret = &worker;
            }

        }

        if (ret != nullptr) {
            SLT_DBG(*ret, "selected worker by lcs similarity, lcs_len = %d, similarity = %f\n", max_lcs_len, max_similarity);
        }
    }

    // find the worker that has been least recently used
    if (ret == nullptr) {
        int64_t t_last = ggml_time_us();
        for (LlamaCppWorker & worker : workers) {
            // skip the worker if it is not available
            if (worker.is_processing()) {
                continue;
            }

            // select the current worker if the criteria match
            if (worker.t_last_used < t_last) {
                t_last = worker.t_last_used;
                ret = &worker;
            }
        }

        if (ret != nullptr) {
            SLT_DBG(*ret, "selected worker by lru, t_last = %" PRId64 "\n", t_last);
        }
    }

    return ret;
}

bool LlamaCppEngine::can_be_detokenized(const struct llama_context * ctx, const std::vector<llama_token> & tokens) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    for (const auto & token : tokens) {
        if (token < 0 || token >= n_vocab) {
            return false;
        }
    }
    return true;
}

bool LlamaCppEngine::launch_worker_with_task(LlamaCppWorker & worker, const ServerTask & task) {
    worker.reset();
    worker.id_task       = task.id;
    worker.index         = task.index;
    worker.task_type     = task.type;
    worker.params        = std::move(task.params);
    worker.prompt_tokens = std::move(task.prompt_tokens);

    if (!are_lora_equal(task.params.lora, worker.lora)) {
        // if lora is changed, we cannot reuse cached tokens
        worker.cache_tokens.clear();
        worker.lora = task.params.lora;
    }

    bool can_detokenize = can_be_detokenized(ctx, worker.prompt_tokens);
    if (!can_detokenize) {
        send_error(task, "Prompt contains invalid tokens", ERROR_TYPE_INVALID_REQUEST);
        return false;
    }
    SLT_DBG(worker, "launching worker : %s\n", safe_json_to_str(worker.to_json()).c_str());

    if (worker.n_predict > 0 && worker.params.n_predict > worker.n_predict) {
        // Might be better to reject the request with a 400 ?
        SLT_WRN(worker, "n_predict = %d exceeds server configuration, setting to %d\n", worker.params.n_predict, worker.n_predict);
        worker.params.n_predict = worker.n_predict;
    }

    if (worker.params.ignore_eos && has_eos_token) {
        worker.params.sampling.logit_bias.push_back({llama_vocab_eos(vocab), -INFINITY});
    }

    {
        if (worker.smpl != nullptr) {
            common_sampler_free(worker.smpl);
        }

        worker.smpl = common_sampler_init(model, worker.params.sampling);
        if (worker.smpl == nullptr) {
            // for now, the only error that may happen here is invalid grammar
            send_error(task, "Failed to parse grammar", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
    }

    if (worker.ctx_dft) {
        llama_batch_free(worker.batch_spec);

        worker.batch_spec = llama_batch_init(worker.params.speculative.n_max + 1, 0, 1);
    }

    worker.state = WORKER_STATE_STARTED;

    SLT_INF(worker, "%s", "processing task\n");

    return true;
}

void LlamaCppEngine::kv_cache_clear() {
    SRV_DBG("%s", "clearing KV cache\n");

    // clear the entire KV cache
    llama_kv_self_clear(ctx);
    clean_kv_cache = false;
}

bool LlamaCppEngine::process_token(CompletionTokenOutput & result, LlamaCppWorker & worker) {
    // remember which tokens were sampled - used for repetition penalties during sampling
    const std::string token_str = result.text_to_send;
    worker.sampled = result.tok;

    worker.generated_text += token_str;
    if (worker.params.return_tokens) {
        worker.generated_tokens.push_back(result.tok);
    }
    worker.has_next_token = true;

    // check if there is incomplete UTF-8 character at the end
    bool incomplete = validate_utf8(worker.generated_text) < worker.generated_text.size();

    // search stop word and delete it
    if (!incomplete) {
        size_t pos = std::min(worker.n_sent_text, worker.generated_text.size());

        const std::string str_test = worker.generated_text.substr(pos);
        bool send_text = true;

        size_t stop_pos = worker.find_stopping_strings(str_test, token_str.size(), true);
        if (stop_pos != std::string::npos) {
            worker.generated_text.erase(
                worker.generated_text.begin() + pos + stop_pos,
                worker.generated_text.end());
            pos = std::min(worker.n_sent_text, worker.generated_text.size());
        } else if (worker.has_next_token) {
            stop_pos = worker.find_stopping_strings(str_test, token_str.size(), false);
            send_text = stop_pos == std::string::npos;
        }

        // check if there is any token to predict
        if (send_text) {
            // no send the stop word in the response
            result.text_to_send = worker.generated_text.substr(pos, std::string::npos);
            worker.n_sent_text += result.text_to_send.size();
            // add the token to worker queue and cache
        } else {
            result.text_to_send = "";
        }

        worker.add_token(result);
        if (worker.params.stream) {
            send_partial_response(worker, result);
        }
    }

    if (incomplete) {
        worker.has_next_token = true;
    }

    // check the limits
    if (worker.n_decoded > 0 && worker.has_next_token && !worker.has_budget(params_base)) {
        worker.stop           = STOP_TYPE_LIMIT;
        worker.has_next_token = false;

        SLT_DBG(worker, "stopped by limit, n_decoded = %d, n_predict = %d\n", worker.n_decoded, worker.params.n_predict);
    }

    if (worker.has_new_line) {
        // require that each new line has a whitespace prefix (i.e. indentation) of at least worker.params.n_indent
        if (worker.params.n_indent > 0) {
            // check the current indentation
            // TODO: improve by not doing it more than once for each new line
            if (worker.last_nl_pos > 0) {
                size_t pos = worker.last_nl_pos;

                int n_indent = 0;
                while (pos < worker.generated_text.size() && (worker.generated_text[pos] == ' ' || worker.generated_text[pos] == '\t')) {
                    n_indent++;
                    pos++;
                }

                if (pos < worker.generated_text.size() && n_indent < worker.params.n_indent) {
                    worker.stop           = STOP_TYPE_LIMIT;
                    worker.has_next_token = false;

                    // cut the last line
                    worker.generated_text.erase(pos, std::string::npos);

                    SLT_DBG(worker, "stopped by indentation limit, n_decoded = %d, n_indent = %d\n", worker.n_decoded, n_indent);
                }
            }

            // find the next new line
            {
                const size_t pos = worker.generated_text.find('\n', worker.last_nl_pos);

                if (pos != std::string::npos) {
                    worker.last_nl_pos = pos + 1;
                }
            }
        }
    }

    // check if there is a new line in the generated text
    if (result.text_to_send.find('\n') != std::string::npos) {
        worker.has_new_line = true;

        // if we have seen a new line, we stop after a certain time limit, but only upon another new line
        if (worker.params.t_max_predict_ms > 0 && (ggml_time_us() - worker.t_start_generation > 1000.0f*worker.params.t_max_predict_ms)) {
            worker.stop           = STOP_TYPE_LIMIT;
            worker.has_next_token = false;

            SLT_DBG(worker, "stopped by time limit, n_decoded = %d, t_max_predict_ms = %d ms\n", worker.n_decoded, (int) worker.params.t_max_predict_ms);
        }
    }

    // if context shift is disabled, we stop when it reaches the context limit
    if (worker.n_past >= worker.n_ctx) {
        worker.truncated      = true;
        worker.stop           = STOP_TYPE_LIMIT;
        worker.has_next_token = false;

        SLT_DBG(worker, "stopped due to running out of context capacity, n_past = %d, n_prompt_tokens = %d, n_decoded = %d, n_ctx = %d\n",
            worker.n_decoded, worker.n_prompt_tokens, worker.n_past, worker.n_ctx);
    }

    if (llama_vocab_is_eog(vocab, result.tok)) {
        worker.stop           = STOP_TYPE_EOS;
        worker.has_next_token = false;

        SLT_DBG(worker, "%s", "stopped by EOS\n");
    }

    const auto n_ctx_train = llama_model_n_ctx_train(model);

    if (worker.params.n_predict < 1 && worker.n_predict < 1 && worker.n_prompt_tokens + worker.n_decoded >= n_ctx_train) {
        worker.truncated      = true;
        worker.stop           = STOP_TYPE_LIMIT;
        worker.has_next_token = false; // stop prediction

        SLT_WRN(worker,
                "n_predict (%d) is set for infinite generation. "
                "Limiting generated tokens to n_ctx_train (%d) to avoid EOS-less generation infinite loop\n",
                worker.params.n_predict, n_ctx_train);
    }

    SLT_DBG(worker, "n_decoded = %d, n_remaining = %d, next token: %5d '%s'\n", worker.n_decoded, worker.n_remaining, result.tok, token_str.c_str());

    return worker.has_next_token; // continue
}

void LlamaCppEngine::populate_token_probs(const LlamaCppWorker & worker, CompletionTokenOutput & result, bool post_sampling, bool special, int idx) {
    size_t n_probs = worker.params.sampling.n_probs;
    size_t n_vocab = llama_vocab_n_tokens(vocab);
    if (post_sampling) {
        const auto * cur_p = common_sampler_get_candidates(worker.smpl);
        const size_t max_probs = cur_p->size;

        // set probability for sampled token
        for (size_t i = 0; i < max_probs; i++) {
            if (cur_p->data[i].id == result.tok) {
                result.prob = cur_p->data[i].p;
                break;
            }
        }

        // set probability for top n_probs tokens
        result.probs.reserve(max_probs);
        for (size_t i = 0; i < std::min(max_probs, n_probs); i++) {
            result.probs.push_back({
                cur_p->data[i].id,
                common_token_to_piece(ctx, cur_p->data[i].id, special),
                cur_p->data[i].p
            });
        }
    } else {
        // TODO: optimize this with min-p optimization
        std::vector<llama_token_data> cur = get_token_probabilities(ctx, idx);

        // set probability for sampled token
        for (size_t i = 0; i < n_vocab; i++) {
            // set probability for sampled token
            if (cur[i].id == result.tok) {
                result.prob = cur[i].p;
                break;
            }
        }

        // set probability for top n_probs tokens
        result.probs.reserve(n_probs);
        for (size_t i = 0; i < std::min(n_vocab, n_probs); i++) {
            result.probs.push_back({
                cur[i].id,
                common_token_to_piece(ctx, cur[i].id, special),
                cur[i].p
            });
        }
    }
}

void LlamaCppEngine::send_error(const ServerTask & task, const std::string & error, const enum error_type type) {
    send_error(task.id, error, type);
}

void LlamaCppEngine::send_error(const LlamaCppWorker & worker, const std::string & error, const enum error_type type) {
    send_error(worker.id_task, error, type);
}

void LlamaCppEngine::send_error(const int id_task, const std::string & error, const enum error_type type) {
    SRV_ERR("task id = %d, error: %s\n", id_task, error.c_str());

    auto res = std::make_unique<ServerTaskResultError>();
    res->id       = id_task;
    res->err_type = type;
    res->err_msg  = error;

    queue_results.send(std::move(res));
}

void LlamaCppEngine::send_partial_response(LlamaCppWorker & worker, const CompletionTokenOutput & tkn) {
    auto res = std::make_unique<ServerTaskResultCmplPartial>();

    res->id      = worker.id_task;
    res->index   = worker.index;
    res->content = tkn.text_to_send;
    res->tokens  = { tkn.tok };

    res->n_decoded           = worker.n_decoded;
    res->n_prompt_tokens     = worker.n_prompt_tokens;
    res->post_sampling_probs = worker.params.post_sampling_probs;

    res->verbose           = worker.params.verbose;
    res->oaicompat         = worker.params.oaicompat;
    res->oaicompat_model   = worker.params.oaicompat_model;
    res->oaicompat_cmpl_id = worker.params.oaicompat_cmpl_id;

    // populate res.probs_output
    if (worker.params.sampling.n_probs > 0) {
        res->prob_output = tkn; // copy the token probs
    }

    // populate timings if this is final response or timings_per_token is enabled
    if (worker.stop != STOP_TYPE_NONE || worker.params.timings_per_token) {
        res->timings = worker.get_timings();
    }

    queue_results.send(std::move(res));
}

void LlamaCppEngine::send_final_response(const LlamaCppWorker & worker) {
    auto res = std::make_unique<ServerTaskResultCmplFinal>();
    res->id              = worker.id_task;
    res->id_worker         = worker.id;

    res->index           = worker.index;
    res->content         = std::move(worker.generated_text);
    res->tokens          = std::move(worker.generated_tokens);
    res->timings         = worker.get_timings();
    res->prompt          = common_detokenize(ctx, worker.prompt_tokens, true);
    res->response_fields = std::move(worker.params.response_fields);

    res->truncated           = worker.truncated;
    res->n_decoded           = worker.n_decoded;
    res->n_prompt_tokens     = worker.n_prompt_tokens;
    res->n_tokens_cached     = worker.n_past;
    res->has_new_line        = worker.has_new_line;
    res->stopping_word       = worker.stopping_word;
    res->stop                = worker.stop;
    res->post_sampling_probs = worker.params.post_sampling_probs;

    res->verbose               = worker.params.verbose;
    res->stream                = worker.params.stream;
    res->oaicompat             = worker.params.oaicompat;
    res->oaicompat_model       = worker.params.oaicompat_model;
    res->oaicompat_cmpl_id     = worker.params.oaicompat_cmpl_id;
    res->oaicompat_chat_format = worker.params.oaicompat_chat_format;
    // populate res.probs_output
    if (worker.params.sampling.n_probs > 0) {
        if (!worker.params.stream && worker.stop == STOP_TYPE_WORD) {
            const llama_tokens stop_word_toks = common_tokenize(ctx, worker.stopping_word, false);

            size_t safe_offset = std::min(worker.generated_token_probs.size(), stop_word_toks.size());
            res->probs_output = std::vector<CompletionTokenOutput>(
                worker.generated_token_probs.begin(),
                worker.generated_token_probs.end() - safe_offset);
        } else {
            res->probs_output = std::vector<CompletionTokenOutput>(
                worker.generated_token_probs.begin(),
                worker.generated_token_probs.end());
        }
    }

    res->generation_params = worker.params; // copy the parameters

    queue_results.send(std::move(res));
}

void LlamaCppEngine::send_embedding(const LlamaCppWorker & worker, const llama_batch & batch) {
    auto res = std::make_unique<ServerTaskResultEmbd>();
    res->id        = worker.id_task;
    res->index     = worker.index;
    res->n_tokens  = worker.n_prompt_tokens;
    res->oaicompat = worker.params.oaicompat;

    const int n_embd = llama_model_n_embd(model);

    std::vector<float> embd_res(n_embd, 0.0f);

    for (int i = 0; i < batch.n_tokens; ++i) {
        if (!batch.logits[i] || batch.seq_id[i][0] != worker.id) {
            continue;
        }

        const float * embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
        if (embd == NULL) {
            embd = llama_get_embeddings_ith(ctx, i);
        }

        if (embd == NULL) {
            SLT_ERR(worker, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

            res->embedding.push_back(std::vector<float>(n_embd, 0.0f));
            continue;
        }

        // normalize only when there is pooling
        // TODO: configurable
        if (llama_pooling_type(worker.ctx) != LLAMA_POOLING_TYPE_NONE) {
            common_embd_normalize(embd, embd_res.data(), n_embd, 2);
            res->embedding.push_back(embd_res);
        } else {
            res->embedding.push_back({ embd, embd + n_embd });
        }
    }

    SLT_DBG(worker, "%s", "sending embeddings\n");

    queue_results.send(std::move(res));
}

void LlamaCppEngine::send_rerank(const LlamaCppWorker & worker, const llama_batch & batch) {
    auto res = std::make_unique<ServerTaskResultRerank>();
    res->id    = worker.id_task;
    res->index = worker.index;
    res->n_tokens = worker.n_prompt_tokens;

    for (int i = 0; i < batch.n_tokens; ++i) {
        if (!batch.logits[i] || batch.seq_id[i][0] != worker.id) {
            continue;
        }

        const float * embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
        if (embd == NULL) {
            embd = llama_get_embeddings_ith(ctx, i);
        }

        if (embd == NULL) {
            SLT_ERR(worker, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

            res->score = -1e6;
            continue;
        }

        res->score = embd[0];
    }

    SLT_DBG(worker, "sending rerank result, res.score = %f\n", res->score);

    queue_results.send(std::move(res));
}

void LlamaCppEngine::cancel_tasks(const std::unordered_set<int> & id_tasks) {
    std::vector<ServerTask> cancel_tasks;
    cancel_tasks.reserve(id_tasks.size());
    for (const auto & id_task : id_tasks) {
        SRV_WRN("cancel task, id_task = %d\n", id_task);

        ServerTask task(SERVER_TASK_TYPE_CANCEL);
        task.id_target = id_task;
        queue_results.remove_waiting_task_id(id_task);
        cancel_tasks.push_back(task);
    }
    // push to beginning of the queue, so it has highest priority
    queue_tasks.post(cancel_tasks, true);
}

void LlamaCppEngine::abort_tasks(const std::unordered_set<int> & id_tasks) {
    std::vector<ServerTask> abort_tasks;
    abort_tasks.reserve(id_tasks.size());
    for (const auto & id_task : id_tasks) {
        SRV_WRN("abort task, id_task = %d\n", id_task);

        ServerTask task(SERVER_TASK_TYPE_ABORT);
        task.id_target = id_task;
        queue_results.remove_waiting_task_id(id_task);
        abort_tasks.push_back(task);
    }
    // push to beginning of the queue, so it has highest priority
    queue_tasks.post(abort_tasks, true);
}

void LlamaCppEngine::receive_multi_results(
    const std::unordered_set<int> & id_tasks,
    const std::function<void(std::vector<ServerTaskResultPtr>&)> & result_handler,
    const std::function<void(json)> & error_handler,
    const std::function<bool()> & is_connection_closed) {

    std::vector<ServerTaskResultPtr> results(id_tasks.size());
    for (int i = 0; i < (int)id_tasks.size(); i++) {
        ServerTaskResultPtr result = queue_results.recv_with_timeout(id_tasks, LLAMA_POLLING_SECONDS);

        if (is_connection_closed()) {
            cancel_tasks(id_tasks);
            return;
        }

        if (result == nullptr) {
            i--; // retry
            continue;
        }

        if (result->is_error()) {
            error_handler(result->to_json());
            cancel_tasks(id_tasks);
            return;
        }

        GGML_ASSERT(
            dynamic_cast<ServerTaskResultCmplFinal*>(result.get()) != nullptr
            || dynamic_cast<ServerTaskResultEmbd*>(result.get()) != nullptr
            || dynamic_cast<ServerTaskResultRerank*>(result.get()) != nullptr
        );
        const size_t idx = result->get_index();
        GGML_ASSERT(idx < results.size() && "index out of range");
        results[idx] = std::move(result);
    }
    result_handler(results);
}

void LlamaCppEngine::receive_cmpl_results_stream(
    const std::unordered_set<int> & id_tasks,
    const std::function<bool(ServerTaskResultPtr&)> & result_handler,
    const std::function<void(json)> & error_handler,
    const std::function<bool()> & is_connection_closed) {

    size_t n_finished = 0;
    while (true) {
        ServerTaskResultPtr result = queue_results.recv_with_timeout(id_tasks, LLAMA_POLLING_SECONDS);

        if (is_connection_closed()) {
            cancel_tasks(id_tasks);
            return;
        }

        if (result == nullptr) {
            continue; // retry
        }

        if (result->is_error()) {
            error_handler(result->to_json());
            cancel_tasks(id_tasks);
            return;
        }

        GGML_ASSERT(
            dynamic_cast<ServerTaskResultCmplPartial*>(result.get()) != nullptr
            || dynamic_cast<ServerTaskResultCmplFinal*>(result.get()) != nullptr
        );
        if (!result_handler(result)) {
            cancel_tasks(id_tasks);
            break;
        }

        if (result->is_stop()) {
            if (++n_finished == id_tasks.size()) {
                break;
            }
        }
    }
}

void LlamaCppEngine::process_single_task(const ServerTask task) {
    switch (task.type) {
        case SERVER_TASK_TYPE_COMPLETION:
        case SERVER_TASK_TYPE_INFILL:
        case SERVER_TASK_TYPE_EMBEDDING:
        case SERVER_TASK_TYPE_RERANK:
            {
                const int id_worker = task.id_selected_worker;

                LlamaCppWorker * worker = id_worker != -1 ? get_worker_by_id(id_worker) : get_available_worker(task);

                if (worker == nullptr) {
                    // if no worker is available, we defer this task for processing later
                    SRV_DBG("no worker is available, defer task, id_task = %d\n", task.id);
                    queue_tasks.defer(task);
                    break;
                }
                if (worker->is_processing()) {
                    // if requested worker is unavailable, we defer this task for processing later
                    SRV_DBG("requested worker is unavailable, defer task, id_task = %d\n", task.id);
                    queue_tasks.defer(task);
                    break;
                }

                if (!launch_worker_with_task(*worker, task)) {
                    SRV_ERR("failed to launch worker with task, id_task = %d\n", task.id);
                    break;
                }
            } break;
        case SERVER_TASK_TYPE_CANCEL:
            {
                // release worker linked with the task id
                for (auto & worker : workers) {
                    if (worker.id_task == task.id_target) {
                        worker.release();
                        break;
                    }
                }
            } break;
        case SERVER_TASK_TYPE_ABORT:
        {
            // set abort flag for worker linked with the task id
            for (auto & worker : workers) {
                if (worker.id_task == task.id_target) {
                    decode_abort_flag_ = true;
                    break;
                }
            }
        } break;
        case SERVER_TASK_TYPE_NEXT_RESPONSE:
            {
                // do nothing
            } break;
        case SERVER_TASK_TYPE_METRICS:
            {
                json workers_data = json::array();

                int n_idle_workers       = 0;
                int n_processing_workers = 0;

                for (LlamaCppWorker & worker : workers) {
                    json worker_data = worker.to_json();

                    if (worker.is_processing()) {
                        n_processing_workers++;
                    } else {
                        n_idle_workers++;
                    }

                    workers_data.push_back(worker_data);
                }
                SRV_DBG("n_idle_workers = %d, n_processing_workers = %d\n", n_idle_workers, n_processing_workers);

                auto res = std::make_unique<ServerTaskResultMetrics>();
                res->id                  = task.id;
                res->workers_data          = std::move(workers_data);
                res->n_idle_workers        = n_idle_workers;
                res->n_processing_workers  = n_processing_workers;
                res->n_tasks_deferred    = queue_tasks.queue_tasks_deferred.size();
                res->t_start             = metrics.t_start;

                res->kv_cache_tokens_count = llama_kv_self_n_tokens(ctx);
                res->kv_cache_used_cells   = llama_kv_self_used_cells(ctx);

                res->n_prompt_tokens_processed_total = metrics.n_prompt_tokens_processed_total;
                res->t_prompt_processing_total       = metrics.t_prompt_processing_total;
                res->n_tokens_predicted_total        = metrics.n_tokens_predicted_total;
                res->t_tokens_generation_total       = metrics.t_tokens_generation_total;

                res->n_prompt_tokens_processed = metrics.n_prompt_tokens_processed;
                res->t_prompt_processing       = metrics.t_prompt_processing;
                res->n_tokens_predicted        = metrics.n_tokens_predicted;
                res->t_tokens_generation       = metrics.t_tokens_generation;

                res->n_decode_total          = metrics.n_decode_total;
                res->n_busy_workers_total      = metrics.n_busy_workers_total;

                if (task.metrics_reset_bucket) {
                    metrics.reset_bucket();
                }
                queue_results.send(std::move(res));
            } break;
        case SERVER_TASK_TYPE_WORKER_SAVE:
            {
                int id_worker = task.worker_action.worker_id;
                LlamaCppWorker * worker = get_worker_by_id(id_worker);
                if (worker == nullptr) {
                    send_error(task, "Invalid worker ID", ERROR_TYPE_INVALID_REQUEST);
                    break;
                }
                if (worker->is_processing()) {
                    // if requested worker is unavailable, we defer this task for processing later
                    SRV_DBG("requested worker is unavailable, defer task, id_task = %d\n", task.id);
                    queue_tasks.defer(task);
                    break;
                }

                const size_t token_count = worker->cache_tokens.size();
                const int64_t t_start = ggml_time_us();

                std::string filename = task.worker_action.filename;
                std::string filepath = task.worker_action.filepath;

                const size_t nwrite = llama_state_seq_save_file(ctx, filepath.c_str(), worker->id, worker->cache_tokens.data(), token_count);

                const int64_t t_end = ggml_time_us();
                const double t_save_ms = (t_end - t_start) / 1000.0;

                auto res = std::make_unique<ServerTaskResultWorkerSaveLoad>();
                res->id       = task.id;
                res->id_worker  = id_worker;
                res->filename = filename;
                res->is_save  = true;
                res->n_tokens = token_count;
                res->n_bytes  = nwrite;
                res->t_ms     = t_save_ms;
                queue_results.send(std::move(res));
            } break;
        case SERVER_TASK_TYPE_WORKER_RESTORE:
            {
                int id_worker = task.worker_action.worker_id;
                LlamaCppWorker * worker = get_worker_by_id(id_worker);
                if (worker == nullptr) {
                    send_error(task, "Invalid worker ID", ERROR_TYPE_INVALID_REQUEST);
                    break;
                }
                if (worker->is_processing()) {
                    // if requested worker is unavailable, we defer this task for processing later
                    SRV_DBG("requested worker is unavailable, defer task, id_task = %d\n", task.id);
                    queue_tasks.defer(task);
                    break;
                }

                const int64_t t_start = ggml_time_us();

                std::string filename = task.worker_action.filename;
                std::string filepath = task.worker_action.filepath;

                worker->cache_tokens.resize(worker->n_ctx);
                size_t token_count = 0;
                size_t nread = llama_state_seq_load_file(ctx, filepath.c_str(), worker->id, worker->cache_tokens.data(), worker->cache_tokens.size(), &token_count);
                if (nread == 0) {
                    worker->cache_tokens.resize(0);
                    send_error(task, "Unable to restore worker, no available space in KV cache or invalid worker save file", ERROR_TYPE_INVALID_REQUEST);
                    break;
                }
                worker->cache_tokens.resize(token_count);
                //restore lora
                worker->lora = params_base.lora_adapters;

                const int64_t t_end = ggml_time_us();
                const double t_restore_ms = (t_end - t_start) / 1000.0;

                auto res = std::make_unique<ServerTaskResultWorkerSaveLoad>();
                res->id       = task.id;
                res->id_worker  = id_worker;
                res->filename = filename;
                res->is_save  = false;
                res->n_tokens = token_count;
                res->n_bytes  = nread;
                res->t_ms     = t_restore_ms;
                queue_results.send(std::move(res));
            } break;
        case SERVER_TASK_TYPE_WORKER_ERASE:
            {
                int id_worker = task.worker_action.worker_id;
                LlamaCppWorker * worker = get_worker_by_id(id_worker);
                if (worker == nullptr) {
                    send_error(task, "Invalid worker ID", ERROR_TYPE_INVALID_REQUEST);
                    break;
                }
                if (worker->is_processing()) {
                    // if requested worker is unavailable, we defer this task for processing later
                    SRV_DBG("requested worker is unavailable, defer task, id_task = %d\n", task.id);
                    queue_tasks.defer(task);
                    break;
                }

                // Erase token cache
                const size_t n_erased = worker->cache_tokens.size();
                llama_kv_self_seq_rm(ctx, worker->id, -1, -1);
                worker->cache_tokens.clear();

                auto res = std::make_unique<ServerTaskResultWorkerErase>();
                res->id       = task.id;
                res->id_worker  = id_worker;
                res->n_erased = n_erased;
                queue_results.send(std::move(res));
            } break;
        case SERVER_TASK_TYPE_SET_LORA:
            {
                params_base.lora_adapters = std::move(task.set_lora);
                auto res = std::make_unique<ServerTaskResultApplyLora>();
                res->id = task.id;
                queue_results.send(std::move(res));
            } break;
    }
}

void LlamaCppEngine::update_workers() {
    // check if all workers are idle
    {
        bool all_idle = true;

        for (auto & worker : workers) {
            if (worker.is_processing()) {
                all_idle = false;
                break;
            }
        }

        if (all_idle) {
            SRV_INF("%s", "all workers are idle\n");
            if (clean_kv_cache) {
                kv_cache_clear();
            }

            return;
        }
    }

    {
        SRV_DBG("%s", "posting NEXT_RESPONSE\n");

        ServerTask task(SERVER_TASK_TYPE_NEXT_RESPONSE);
        task.id = queue_tasks.get_new_id();
        queue_tasks.post(task);
    }

    // apply context-shift if needed
    // TODO: simplify and improve
    for (LlamaCppWorker & worker : workers) {
        if (worker.is_processing() && worker.n_past + 1 >= worker.n_ctx) {
            SLT_DBG(worker, "try kv_self_expansion, n_past = %d, worker.n_ctx = %d\n", worker.n_past, worker.n_ctx);
            if(llama_kv_self_expansion(ctx, 1)) {
                n_ctx = llama_n_ctx(ctx);
                for (auto& w : workers) {
                    w.n_ctx = n_ctx / params_base.n_parallel;
                }
            } else {
                if (!params_base.ctx_shift) {
                    // this check is redundant (for good)
                    // we should never get here, because generation should already stopped in process_token()
                    worker.release();
                    send_error(worker, "context shift is disabled", ERROR_TYPE_SERVER);
                    continue;
                }

                // Shift context
                const int n_keep    = worker.params.n_keep + add_bos_token;

                int32_t n_discard = llama_kv_self_shift(ctx, worker.id, worker.n_past, n_keep);
                SLT_WRN(worker, "slot context shift, n_keep = %d, n_discard = %d\n", n_keep, n_discard);

                if (worker.params.cache_prompt) {
                    for (size_t i = n_keep + n_discard; i < worker.cache_tokens.size(); i++) {
                        worker.cache_tokens[i - n_discard] = worker.cache_tokens[i];
                    }

                    worker.cache_tokens.resize(worker.cache_tokens.size() - n_discard);
                }

                worker.n_past -= n_discard;

                worker.truncated = true;
            }
        }
    }

    // start populating the batch for this iteration
    common_batch_clear(batch);

    // track if given worker can be batched with workers already in the batch
    LlamaCppWorker * worker_batched = nullptr;

    auto accept_special_token = [&](LlamaCppWorker & worker, llama_token token) {
        return params_base.special || worker.params.sampling.preserved_tokens.find(token) != worker.params.sampling.preserved_tokens.end();
    };

    // frist, add sampled tokens from any ongoing sequences
    for (auto & worker : workers) {
        if (worker.state != WORKER_STATE_GENERATING) {
            continue;
        }

        // check if we can batch this worker with the previous one
        if (!worker_batched) {
            worker_batched = &worker;
        } else if (!worker_batched->can_batch_with(worker)) {
            continue;
        }

        worker.i_batch = batch.n_tokens;

        common_batch_add(batch, worker.sampled, worker.n_past, { worker.id }, true);

        worker.n_past += 1;

        if (worker.params.cache_prompt) {
            worker.cache_tokens.push_back(worker.sampled);
        }

        SLT_DBG(worker, "worker decode token, n_ctx = %d, n_past = %d, n_cache_tokens = %d, truncated = %d\n",
                worker.n_ctx, worker.n_past, (int) worker.cache_tokens.size(), worker.truncated);
    }

    // process in chunks of params.n_batch
    int32_t n_batch  = llama_n_batch(ctx);
    int32_t n_ubatch = llama_n_ubatch(ctx);

    // next, batch any pending prompts without exceeding n_batch
    if (params_base.cont_batching || batch.n_tokens == 0) {
        for (auto & worker : workers) {
            // check if we can batch this worker with the previous one
            if (worker.is_processing()) {
                if (!worker_batched) {
                    worker_batched = &worker;
                } else if (!worker_batched->can_batch_with(worker)) {
                    continue;
                }
            }

            // this worker still has a prompt to be processed
            if (worker.state == WORKER_STATE_PROCESSING_PROMPT || worker.state == WORKER_STATE_STARTED) {
                auto & prompt_tokens = worker.prompt_tokens;

                // TODO: maybe move branch to outside of this loop in the future
                if (worker.state == WORKER_STATE_STARTED) {
                    worker.t_start_process_prompt = ggml_time_us();
                    worker.t_start_generation = 0;

                    worker.n_past = 0;
                    worker.n_prompt_tokens = prompt_tokens.size();
                    worker.state = WORKER_STATE_PROCESSING_PROMPT;

                    SLT_INF(worker, "new prompt, n_ctx_worker = %d, n_keep = %d, n_prompt_tokens = %d\n", worker.n_ctx, worker.params.n_keep, worker.n_prompt_tokens);

                    // print prompt tokens (for debugging)
                    if (1) {
                        // first 16 tokens (avoid flooding logs)
                        for (int i = 0; i < std::min<int>(16, prompt_tokens.size()); i++) {
                            SLT_DBG(worker, "prompt token %3d: %6d '%s'\n", i, prompt_tokens[i], common_token_to_piece(ctx, prompt_tokens[i]).c_str());
                        }
                    } else {
                        // all
                        for (int i = 0; i < (int) prompt_tokens.size(); i++) {
                            SLT_DBG(worker, "prompt token %3d: %6d '%s'\n", i, prompt_tokens[i], common_token_to_piece(ctx, prompt_tokens[i]).c_str());
                        }
                    }

                    // empty prompt passed -> release the worker and send empty response
                    if (prompt_tokens.empty()) {
                        SLT_WRN(worker, "%s", "empty prompt - releasing worker\n");

                        worker.release();
                        worker.print_timings();
                        send_final_response(worker);
                        continue;
                    }

                    if (worker.is_non_causal()) {
                        if (worker.n_prompt_tokens > n_ubatch) {
                            worker.release();
                            send_error(worker, "input is too large to process. increase the physical batch size", ERROR_TYPE_SERVER);
                            continue;
                        }

                        if (worker.n_prompt_tokens > worker.n_ctx) {
                            worker.release();
                            send_error(worker, "input is larger than the max context size. skipping", ERROR_TYPE_SERVER);
                            continue;
                        }
                    } else {
                        if (!params_base.ctx_shift) {
                            // if context shift is disabled, we make sure prompt size is smaller than KV size
                            // TODO: there should be a separate parameter that control prompt truncation
                            //       context shift should be applied only during the generation phase
                            if (worker.n_prompt_tokens >= worker.n_ctx) {
                                worker.release();
                                send_error(worker, "the request exceeds the available context size. try increasing the context size or enable context shift", ERROR_TYPE_INVALID_REQUEST);
                                continue;
                            }
                        }
                        if (worker.params.n_keep < 0) {
                            worker.params.n_keep = worker.n_prompt_tokens;
                        }
                        worker.params.n_keep = std::min(worker.n_ctx - 4, worker.params.n_keep);

                        // if input prompt is too big, truncate it
                        if (worker.n_prompt_tokens >= worker.n_ctx) {
                            SLT_DBG(worker, "try prompt kv_self_expansion, n_prompt_tokens = %d, worker.n_ctx = %d\n", worker.n_prompt_tokens, worker.n_ctx);
                            if(llama_kv_self_expansion(ctx, worker.n_prompt_tokens)) {
                                n_ctx = llama_n_ctx(ctx);
                                for (auto& w : workers) {
                                    w.n_ctx = n_ctx / params_base.n_parallel;
                                }
                            } else {
                                const int n_left = worker.n_ctx - worker.params.n_keep;

                                const int n_block_size = n_left / 2;
                                const int erased_blocks = (worker.n_prompt_tokens - worker.params.n_keep - n_block_size) / n_block_size;

                                llama_tokens new_tokens(
                                        prompt_tokens.begin(),
                                        prompt_tokens.begin() + worker.params.n_keep);

                                new_tokens.insert(
                                        new_tokens.end(),
                                        prompt_tokens.begin() + worker.params.n_keep + erased_blocks * n_block_size,
                                        prompt_tokens.end());

                                prompt_tokens = std::move(new_tokens);

                                worker.truncated = true;
                                worker.n_prompt_tokens = prompt_tokens.size();

                                SLT_WRN(worker, "input truncated, n_ctx = %d, n_keep = %d, n_left = %d, n_prompt_tokens = %d\n", worker.n_ctx, worker.params.n_keep, n_left, worker.n_prompt_tokens);

                                GGML_ASSERT(worker.n_prompt_tokens < worker.n_ctx);
                            }
                        }

                        if (worker.params.cache_prompt) {
                            // reuse any previously computed tokens that are common with the new prompt
                            worker.n_past = common_lcp(worker.cache_tokens, prompt_tokens);

                            // reuse chunks from the cached prompt by shifting their KV cache in the new position
                            if (params_base.n_cache_reuse > 0) {
                                size_t head_c = worker.n_past; // cache
                                size_t head_p = worker.n_past; // current prompt

                                SLT_DBG(worker, "trying to reuse chunks with size > %d, worker.n_past = %d\n", params_base.n_cache_reuse, worker.n_past);

                                while (head_c < worker.cache_tokens.size() &&
                                       head_p < prompt_tokens.size()) {

                                    size_t n_match = 0;
                                    while (head_c + n_match < worker.cache_tokens.size() &&
                                           head_p + n_match < prompt_tokens.size()     &&
                                           worker.cache_tokens[head_c + n_match] == prompt_tokens[head_p + n_match]) {

                                        n_match++;
                                    }

                                    if (n_match >= (size_t) params_base.n_cache_reuse) {
                                        SLT_INF(worker, "reusing chunk with size %zu, shifting KV cache [%zu, %zu) -> [%zu, %zu)\n", n_match, head_c, head_c + n_match, head_p, head_p + n_match);
                                        //for (size_t i = head_p; i < head_p + n_match; i++) {
                                        //    SLT_DBG(worker, "cache token %3zu: %6d '%s'\n", i, prompt_tokens[i], common_token_to_piece(ctx, prompt_tokens[i]).c_str());
                                        //}

                                        const int64_t kv_shift = (int64_t) head_p - (int64_t) head_c;

                                        llama_kv_self_seq_rm (ctx, worker.id, head_p, head_c);
                                        llama_kv_self_seq_add(ctx, worker.id, head_c, head_c + n_match, kv_shift);

                                        for (size_t i = 0; i < n_match; i++) {
                                            worker.cache_tokens[head_p + i] = worker.cache_tokens[head_c + i];
                                            worker.n_past++;
                                        }

                                        head_c += n_match;
                                        head_p += n_match;
                                    } else {
                                        head_c += 1;
                                    }
                                }

                                SLT_DBG(worker, "after context reuse, new worker.n_past = %d\n", worker.n_past);
                            }
                        }
                    }

                    if (worker.n_past == worker.n_prompt_tokens && worker.n_past > 0) {
                        // we have to evaluate at least 1 token to generate logits.
                        SLT_WRN(worker, "need to evaluate at least 1 token to generate logits, n_past = %d, n_prompt_tokens = %d\n", worker.n_past, worker.n_prompt_tokens);

                        worker.n_past--;
                    }

                    worker.n_prompt_tokens_processed = 0;
                }

                // non-causal tasks require to fit the entire prompt in the physical batch
                if (worker.is_non_causal()) {
                    // cannot fit the prompt in the current batch - will try next iter
                    if (batch.n_tokens + worker.n_prompt_tokens > n_batch) {
                        continue;
                    }
                }

                // keep only the common part
                if (!llama_kv_self_seq_rm(ctx, worker.id, worker.n_past, -1)) {
                    // could not partially delete (likely using a non-Transformer model)
                    llama_kv_self_seq_rm(ctx, worker.id, -1, -1);

                    // there is no common part left
                    worker.n_past = 0;
                }

                SLT_INF(worker, "kv cache rm [%d, end)\n", worker.n_past);

                // remove the non-common part from the cache
                worker.cache_tokens.resize(worker.n_past);

                // add prompt tokens for processing in the current batch
                while (worker.n_past < worker.n_prompt_tokens && batch.n_tokens < n_batch) {
                    // without pooling, we want to output the embeddings for all the tokens in the batch
                    const bool need_embd = worker.task_type == SERVER_TASK_TYPE_EMBEDDING && llama_pooling_type(worker.ctx) == LLAMA_POOLING_TYPE_NONE;

                    common_batch_add(batch, prompt_tokens[worker.n_past], worker.n_past, { worker.id }, need_embd);

                    if (worker.params.cache_prompt) {
                        worker.cache_tokens.push_back(prompt_tokens[worker.n_past]);
                    }

                    worker.n_prompt_tokens_processed++;
                    worker.n_past++;
                }

                SLT_INF(worker, "prompt processing progress, n_past = %d, n_tokens = %d, progress = %f\n", worker.n_past, batch.n_tokens, (float) worker.n_prompt_tokens_processed / worker.n_prompt_tokens);

                // entire prompt has been processed
                if (worker.n_past == worker.n_prompt_tokens) {
                    worker.state = WORKER_STATE_DONE_PROMPT;

                    GGML_ASSERT(batch.n_tokens > 0);

                    common_sampler_reset(worker.smpl);

                    // Process all prompt tokens through sampler system
                    for (int i = 0; i < worker.n_prompt_tokens; ++i) {
                        common_sampler_accept(worker.smpl, prompt_tokens[i], false);
                    }

                    // extract the logits only for the last token
                    batch.logits[batch.n_tokens - 1] = true;

                    worker.n_decoded = 0;
                    worker.i_batch   = batch.n_tokens - 1;

                    SLT_INF(worker, "prompt done, n_past = %d, n_tokens = %d\n", worker.n_past, batch.n_tokens);
                }
            }

            if (batch.n_tokens >= n_batch) {
                break;
            }
        }
    }

    if (batch.n_tokens == 0) {
        SRV_WRN("%s", "no tokens to decode\n");
        return;
    }

    if (worker_batched) {
        // make sure we're in the right embedding mode
        llama_set_embeddings(ctx, worker_batched->is_non_causal());
        // apply lora, only need to do it once per batch
        common_set_adapter_lora(ctx, worker_batched->lora);
    }

    // process the created batch of tokens
    for (int32_t i = 0; i < batch.n_tokens; i += n_batch) {
        const int32_t n_tokens = std::min(n_batch, batch.n_tokens - i);

        llama_batch batch_view = {
            n_tokens,
            batch.token    + i,
            nullptr,
            batch.pos      + i,
            batch.n_seq_id + i,
            batch.seq_id   + i,
            batch.logits   + i,
        };

        const int ret = llama_decode(ctx, batch_view);

        metrics.on_decoded(workers);

        if (ret != 0) {
            if (n_batch == 1 || ret < 0) {
                // if you get here, it means the KV cache is full - try increasing it via the context size
                SRV_ERR("failed to decode the batch: KV cache is full - try increasing it via the context size, i = %d, n_batch = %d, ret = %d\n", i, n_batch, ret);
                for (auto & worker : workers) {
                    worker.release();
                    send_error(worker, "Input prompt is too big compared to KV size. Please try increasing KV size.");
                }
                break; // break loop of n_batch
            }

            if (ret == 2) {
                for (auto & worker : workers) {
                    worker.cache_tokens.clear();
                    worker.release();
                    send_error(worker, "decode abort.", ERROR_TYPE_DECODE_ABORT);
                }
                break; // break loop of n_batch
            }

            // retry with half the batch size to try to find a free worker in the KV cache
            n_batch /= 2;
            i -= n_batch;

            SRV_WRN("failed to find free space in the KV cache, retrying with smaller batch size - try increasing it via the context size or enable defragmentation, i = %d, n_batch = %d, ret = %d\n", i, n_batch, ret);

            continue; // continue loop of n_batch
        }

        for (auto & worker : workers) {
            if (worker.i_batch < (int) i || worker.i_batch >= (int) (i + n_tokens)) {
                continue; // continue loop of workers
            }

            if (worker.state == WORKER_STATE_DONE_PROMPT) {
                if (worker.task_type == SERVER_TASK_TYPE_EMBEDDING) {
                    // prompt evaluated for embedding
                    send_embedding(worker, batch_view);
                    worker.release();
                    worker.i_batch = -1;
                    continue; // continue loop of workers
                }

                if (worker.task_type == SERVER_TASK_TYPE_RERANK) {
                    send_rerank(worker, batch_view);
                    worker.release();
                    worker.i_batch = -1;
                    continue; // continue loop of workers
                }

                // prompt evaluated for next-token prediction
                worker.state = WORKER_STATE_GENERATING;
            } else if (worker.state != WORKER_STATE_GENERATING) {
                continue; // continue loop of workers
            }

            const int tok_idx = worker.i_batch - i;

            llama_token id = common_sampler_sample(worker.smpl, ctx, tok_idx);

            worker.i_batch = -1;

            common_sampler_accept(worker.smpl, id, true);

            worker.n_decoded += 1;

            const int64_t t_current = ggml_time_us();

            if (worker.n_decoded == 1) {
                worker.t_start_generation = t_current;
                worker.t_prompt_processing = (worker.t_start_generation - worker.t_start_process_prompt) / 1e3;
                metrics.on_prompt_eval(worker);
            }

            worker.t_token_generation = (t_current - worker.t_start_generation) / 1e3;

            CompletionTokenOutput result;
            result.tok          = id;
            result.text_to_send = common_token_to_piece(ctx, result.tok, accept_special_token(worker, result.tok));
            result.prob         = 1.0f; // TODO: set it here instead of doing inside populate_token_probs

            if (worker.params.sampling.n_probs > 0) {
                populate_token_probs(worker, result, worker.params.post_sampling_probs, params_base.special, tok_idx);
            }

            if (!process_token(result, worker)) {
                // release worker because of stop condition
                worker.release();
                worker.print_timings();
                send_final_response(worker);
                metrics.on_prediction(worker);
                continue;
            }
        }

        // do speculative decoding
        for (auto & worker : workers) {
            if (!worker.is_processing() || !worker.can_speculate()) {
                continue;
            }

            if (worker.state != WORKER_STATE_GENERATING) {
                continue;
            }

            // determine the max draft that fits the current worker state
            int n_draft_max = worker.params.speculative.n_max;

            // note: n_past is not yet increased for the `id` token sampled above
            //       also, need to leave space for 1 extra token to allow context shifts
            n_draft_max = std::min(n_draft_max, worker.n_ctx - worker.n_past - 2);

            if (worker.n_remaining > 0) {
                n_draft_max = std::min(n_draft_max, worker.n_remaining - 1);
            }

            SLT_DBG(worker, "max possible draft: %d\n", n_draft_max);

            if (n_draft_max < worker.params.speculative.n_min) {
                SLT_DBG(worker, "the max possible draft is too small: %d < %d - skipping speculative decoding\n", n_draft_max, worker.params.speculative.n_min);

                continue;
            }

            llama_token id = worker.sampled;

            struct common_speculative_params params_spec;
            params_spec.n_draft   = n_draft_max;
            params_spec.n_reuse   = llama_n_ctx(worker.ctx_dft) - worker.params.speculative.n_max;
            params_spec.p_min     = worker.params.speculative.p_min;

            llama_tokens draft = common_speculative_gen_draft(worker.spec, params_spec, worker.cache_tokens, id);

            // ignore small drafts
            if (worker.params.speculative.n_min > (int) draft.size()) {
                SLT_DBG(worker, "ignoring small draft: %d < %d\n", (int) draft.size(), worker.params.speculative.n_min);

                continue;
            }

            // construct the speculation batch
            common_batch_clear(worker.batch_spec);
            common_batch_add  (worker.batch_spec, id, worker.n_past, { worker.id }, true);

            for (size_t i = 0; i < draft.size(); ++i) {
                common_batch_add(worker.batch_spec, draft[i], worker.n_past + 1 + i, { worker.id }, true);
            }

            SLT_DBG(worker, "decoding speculative batch, size = %d\n", worker.batch_spec.n_tokens);

            llama_decode(ctx, worker.batch_spec);

            // the accepted tokens from the speculation
            const auto ids = common_sampler_sample_and_accept_n(worker.smpl, ctx, draft);

            worker.n_past    += ids.size();
            worker.n_decoded += ids.size();

            worker.cache_tokens.push_back(id);
            worker.cache_tokens.insert(worker.cache_tokens.end(), ids.begin(), ids.end() - 1);

            llama_kv_self_seq_rm(ctx, worker.id, worker.n_past, -1);

            for (size_t i = 0; i < ids.size(); ++i) {
                CompletionTokenOutput result;

                result.tok          = ids[i];
                result.text_to_send = common_token_to_piece(ctx, result.tok, accept_special_token(worker, result.tok));
                result.prob         = 1.0f; // set later

                // TODO: set result.probs

                if (!process_token(result, worker)) {
                    // release worker because of stop condition
                    worker.release();
                    worker.print_timings();
                    send_final_response(worker);
                    metrics.on_prediction(worker);
                    break;
                }
            }

            SLT_DBG(worker, "accepted %d/%d draft tokens, new n_past = %d\n", (int) ids.size() - 1, (int) draft.size(), worker.n_past);
        }
    }

    SRV_DBG("%s", "run workers completed\n");
}

json LlamaCppEngine::model_meta() const {
    return json {
        {"vocab_type",  llama_vocab_type       (vocab)},
        {"n_vocab",     llama_vocab_n_tokens   (vocab)},
        {"n_ctx_train", llama_model_n_ctx_train(model)},
        {"n_embd",      llama_model_n_embd     (model)},
        {"n_params",    llama_model_n_params   (model)},
        {"size",        llama_model_size       (model)},
    };
}

} // namespace triton::backend::llamacpp