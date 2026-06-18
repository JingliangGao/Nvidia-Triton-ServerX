#include "llamacpp_worker.h"
#include "llamacpp_task.h"
#include "llamacpp_utils.h"

namespace triton::backend::llamacpp
{

json WorkerParams::to_json() const {
    std::vector<std::string> samplers;
    samplers.reserve(sampling.samplers.size());
    for (const auto & sampler : sampling.samplers) {
        samplers.emplace_back(common_sampler_type_to_str(sampler));
    }

    json lora = json::array();
    for (size_t i = 0; i < this->lora.size(); ++i) {
        lora.push_back({{"id", i}, {"scale", this->lora[i].scale}});
    }

    auto grammar_triggers = json::array();
    for (const auto & trigger : sampling.grammar_triggers) {
        grammar_triggers.push_back(trigger.to_json<json>());
    }

    return json {
        {"n_predict",                 n_predict},     // Server configured n_predict
        {"seed",                      sampling.seed},
        {"temperature",               sampling.temp},
        {"dynatemp_range",            sampling.dynatemp_range},
        {"dynatemp_exponent",         sampling.dynatemp_exponent},
        {"top_k",                     sampling.top_k},
        {"top_p",                     sampling.top_p},
        {"min_p",                     sampling.min_p},
        {"xtc_probability",           sampling.xtc_probability},
        {"xtc_threshold",             sampling.xtc_threshold},
        {"typical_p",                 sampling.typ_p},
        {"repeat_last_n",             sampling.penalty_last_n},
        {"repeat_penalty",            sampling.penalty_repeat},
        {"presence_penalty",          sampling.penalty_present},
        {"frequency_penalty",         sampling.penalty_freq},
        {"dry_multiplier",            sampling.dry_multiplier},
        {"dry_base",                  sampling.dry_base},
        {"dry_allowed_length",        sampling.dry_allowed_length},
        {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
        {"dry_sequence_breakers",     sampling.dry_sequence_breakers},
        {"mirostat",                  sampling.mirostat},
        {"mirostat_tau",              sampling.mirostat_tau},
        {"mirostat_eta",              sampling.mirostat_eta},
        {"stop",                      antiprompt},
        {"max_tokens",                n_predict}, // User configured n_predict
        {"n_keep",                    n_keep},
        {"n_discard",                 n_discard},
        {"ignore_eos",                sampling.ignore_eos},
        {"stream",                    stream},
        {"logit_bias",                format_logit_bias(sampling.logit_bias)},
        {"n_probs",                   sampling.n_probs},
        {"min_keep",                  sampling.min_keep},
        {"grammar",                   sampling.grammar},
        {"grammar_lazy",              sampling.grammar_lazy},
        {"grammar_triggers",          grammar_triggers},
        {"preserved_tokens",          sampling.preserved_tokens},
        {"chat_format",               common_chat_format_name(oaicompat_chat_format)},
        {"samplers",                  samplers},
        {"speculative.n_max",         speculative.n_max},
        {"speculative.n_min",         speculative.n_min},
        {"speculative.p_min",         speculative.p_min},
        {"timings_per_token",         timings_per_token},
        {"post_sampling_probs",       post_sampling_probs},
        {"lora",                      lora},
    };
}

void LlamaCppWorker::reset() {
    SLT_DBG(*this, "%s", "\n");

    n_prompt_tokens    = 0;
    last_nl_pos        = 0;
    generated_text     = "";
    has_new_line       = false;
    truncated          = false;
    stop               = STOP_TYPE_NONE;
    stopping_word      = "";
    n_past             = 0;
    n_sent_text        = 0;
    task_type          = SERVER_TASK_TYPE_COMPLETION;

    generated_tokens.clear();
    generated_token_probs.clear();
}

bool LlamaCppWorker::has_budget(const common_params & global_params) {
    if (params.n_predict == -1 && global_params.n_predict == -1) {
        return true; // limitless
    }

    n_remaining = -1;

    if (params.n_predict != -1) {
        n_remaining = params.n_predict - n_decoded;
    } else if (global_params.n_predict != -1) {
        n_remaining = global_params.n_predict - n_decoded;
    }

    return n_remaining > 0; // no budget
}

bool LlamaCppWorker::is_processing() const {
    return state != WORKER_STATE_IDLE;
}

bool LlamaCppWorker::can_speculate() const {
    return ctx_dft && params.speculative.n_max > 0 && params.cache_prompt;
}

void LlamaCppWorker::add_token(const CompletionTokenOutput & token) {
    if (!is_processing()) {
        SLT_WRN(*this, "%s", "worker is not processing\n");
        return;
    }
    generated_token_probs.push_back(token);
}

void LlamaCppWorker::release() {
    if (is_processing()) {
        SLT_INF(*this, "stop processing: n_past = %d, truncated = %d\n", n_past, truncated);

        t_last_used = ggml_time_us();
        t_token_generation = (ggml_time_us() - t_start_generation) / 1e3;
        state = WORKER_STATE_IDLE;
        callback_on_release(id);
    }
}

ResultTimings LlamaCppWorker::get_timings() const {
    ResultTimings timings;
    timings.prompt_n = n_prompt_tokens_processed;
    timings.prompt_ms = t_prompt_processing;
    timings.prompt_per_token_ms = t_prompt_processing / n_prompt_tokens_processed;
    timings.prompt_per_second = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

    timings.predicted_n = n_decoded;
    timings.predicted_ms = t_token_generation;
    timings.predicted_per_token_ms = t_token_generation / n_decoded;
    timings.predicted_per_second = 1e3 / t_token_generation * n_decoded;

    return timings;
}

size_t LlamaCppWorker::find_stopping_strings(const std::string & text, const size_t last_token_size, bool is_full_stop) {
    size_t stop_pos = std::string::npos;

    for (const std::string & word : params.antiprompt) {
        size_t pos;

        if (is_full_stop) {
            const size_t tmp      = word.size() + last_token_size;
            const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

            pos = text.find(word, from_pos);
        } else {
            // otherwise, partial stop
            pos = find_partial_stop_string(word, text);
        }

        if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
            if (is_full_stop) {
                stop           = STOP_TYPE_WORD;
                stopping_word  = word;
                has_next_token = false;
            }
            stop_pos = pos;
        }
    }

    return stop_pos;
}

void LlamaCppWorker::print_timings() const {
    const double t_prompt        =       t_prompt_processing / n_prompt_tokens_processed;
    const double n_prompt_second = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

    const double t_gen        =       t_token_generation / n_decoded;
    const double n_gen_second = 1e3 / t_token_generation * n_decoded;

    SLT_INF(*this,
            "\n"
            "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n"
            "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n"
            "      total time = %10.2f ms / %5d tokens\n",
            t_prompt_processing, n_prompt_tokens_processed, t_prompt, n_prompt_second,
            t_token_generation, n_decoded, t_gen, n_gen_second,
            t_prompt_processing + t_token_generation, n_prompt_tokens_processed + n_decoded);
}

json LlamaCppWorker::to_json() const {
    return json {
        {"id",            id},
        {"id_task",       id_task},
        {"n_ctx",         n_ctx},
        {"speculative",   can_speculate()},
        {"is_processing", is_processing()},
        {"non_causal",    is_non_causal()},
        {"params",        params.to_json()},
        {"prompt",        common_detokenize(ctx, prompt_tokens)},
        {"next_token",
            {
                {"has_next_token", has_next_token},
                {"has_new_line",   has_new_line},
                {"n_remain",       n_remaining},
                {"n_decoded",      n_decoded},
                {"stopping_word",  stopping_word},
            }
        },
    };
}

} //triton::backend::llamacpp