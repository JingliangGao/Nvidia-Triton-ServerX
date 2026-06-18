#include "llamacpp_task.h"
#include "llamacpp_worker.h"

#include <llamacpp/json-schema-to-grammar.h>

#include <limits>

namespace triton::backend::llamacpp
{

inline std::string stop_type_to_str(StopType type) {
    switch (type) {
        case STOP_TYPE_EOS:   return "eos";
        case STOP_TYPE_WORD:  return "word";
        case STOP_TYPE_LIMIT: return "limit";
        default:              return "none";
    }
}

static json format_error_response(const std::string & message, const enum error_type type) {
    std::string type_str;
    int code = 500;
    switch (type) {
        case ERROR_TYPE_INVALID_REQUEST:
            type_str = "invalid_request_error";
            code = 400;
            break;
        case ERROR_TYPE_AUTHENTICATION:
            type_str = "authentication_error";
            code = 401;
            break;
        case ERROR_TYPE_NOT_FOUND:
            type_str = "not_found_error";
            code = 404;
            break;
        case ERROR_TYPE_SERVER:
            type_str = "server_error";
            code = 500;
            break;
        case ERROR_TYPE_PERMISSION:
            type_str = "permission_error";
            code = 403;
            break;
        case ERROR_TYPE_NOT_SUPPORTED:
            type_str = "not_supported_error";
            code = 501;
            break;
        case ERROR_TYPE_UNAVAILABLE:
            type_str = "unavailable_error";
            code = 503;
            break;
        case ERROR_TYPE_DECODE_ABORT:
            type_str = "decode_abort_error";
            code = 504; // 503 is used for decode abort errors to indicate that the
    }
    return json {
        {"code", code},
        {"message", message},
        {"type", type_str},
    };
}

WorkerParams ServerTask::params_from_json_cmpl(
        const llama_context * ctx,
        const common_params & params_base,
        const json & data) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    WorkerParams params;

    // Sampling parameter defaults are loaded from the global server context (but individual requests can still override them)
    WorkerParams defaults;
    defaults.sampling    = params_base.sampling;
    defaults.speculative = params_base.speculative;

    // enabling this will output extra debug information in the HTTP responses from the server
    params.verbose           = params_base.verbosity > 9;
    params.timings_per_token = json_value(data, "timings_per_token", false);

    params.stream           = json_value(data, "stream",             false);
    params.cache_prompt     = json_value(data, "cache_prompt",       true);
    params.return_tokens    = json_value(data, "return_tokens",      false);
    params.n_predict        = json_value(data, "n_predict",          json_value(data, "max_tokens", defaults.n_predict));
    params.n_indent         = json_value(data, "n_indent",           defaults.n_indent);
    params.n_keep           = json_value(data, "n_keep",             defaults.n_keep);
    params.n_discard        = json_value(data, "n_discard",          defaults.n_discard);
    //params.t_max_prompt_ms  = json_value(data, "t_max_prompt_ms",    defaults.t_max_prompt_ms); // TODO: implement
    params.t_max_predict_ms = json_value(data, "t_max_predict_ms",   defaults.t_max_predict_ms);
    params.response_fields  = json_value(data, "response_fields",   std::vector<std::string>());

    params.sampling.top_k              = json_value(data, "top_k",              defaults.sampling.top_k);
    params.sampling.top_p              = json_value(data, "top_p",              defaults.sampling.top_p);
    params.sampling.min_p              = json_value(data, "min_p",              defaults.sampling.min_p);
    params.sampling.xtc_probability    = json_value(data, "xtc_probability",    defaults.sampling.xtc_probability);
    params.sampling.xtc_threshold      = json_value(data, "xtc_threshold",      defaults.sampling.xtc_threshold);
    params.sampling.typ_p              = json_value(data, "typical_p",          defaults.sampling.typ_p);
    params.sampling.temp               = json_value(data, "temperature",        defaults.sampling.temp);
    params.sampling.dynatemp_range     = json_value(data, "dynatemp_range",     defaults.sampling.dynatemp_range);
    params.sampling.dynatemp_exponent  = json_value(data, "dynatemp_exponent",  defaults.sampling.dynatemp_exponent);
    params.sampling.penalty_last_n     = json_value(data, "repeat_last_n",      defaults.sampling.penalty_last_n);
    params.sampling.penalty_repeat     = json_value(data, "repeat_penalty",     defaults.sampling.penalty_repeat);
    params.sampling.penalty_freq       = json_value(data, "frequency_penalty",  defaults.sampling.penalty_freq);
    params.sampling.penalty_present    = json_value(data, "presence_penalty",   defaults.sampling.penalty_present);
    params.sampling.dry_multiplier     = json_value(data, "dry_multiplier",     defaults.sampling.dry_multiplier);
    params.sampling.dry_base           = json_value(data, "dry_base",           defaults.sampling.dry_base);
    params.sampling.dry_allowed_length = json_value(data, "dry_allowed_length", defaults.sampling.dry_allowed_length);
    params.sampling.dry_penalty_last_n = json_value(data, "dry_penalty_last_n", defaults.sampling.dry_penalty_last_n);
    params.sampling.mirostat           = json_value(data, "mirostat",           defaults.sampling.mirostat);
    params.sampling.mirostat_tau       = json_value(data, "mirostat_tau",       defaults.sampling.mirostat_tau);
    params.sampling.mirostat_eta       = json_value(data, "mirostat_eta",       defaults.sampling.mirostat_eta);
    params.sampling.seed               = json_value(data, "seed",               defaults.sampling.seed);
    params.sampling.n_probs            = json_value(data, "n_probs",            defaults.sampling.n_probs);
    params.sampling.min_keep           = json_value(data, "min_keep",           defaults.sampling.min_keep);
    params.post_sampling_probs         = json_value(data, "post_sampling_probs", defaults.post_sampling_probs);

    // reasoning-budget sampler (common/reasoning-budget); defaults match llama-cli. Optional: reasoning_budget_tag_*.
    {
        auto it = data.find("reasoning_budget");
        if (it != data.end() && !it->is_null()) {
            if (!it->is_number()) {
                throw std::runtime_error("reasoning_budget must be a number");
            }
            const int64_t rb64 = it->get<int64_t>();
            if (rb64 < static_cast<int64_t>(std::numeric_limits<int32_t>::min()) ||
                rb64 > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
                throw std::runtime_error("reasoning_budget out of range");
            }
            const int32_t rb = static_cast<int32_t>(rb64);
            if (rb >= 0) {
                const std::string think_start = json_value(
                    data, "reasoning_budget_tag_start", std::string("<think>"));
                const std::string think_end = json_value(
                    data, "reasoning_budget_tag_end", std::string("</think>"));
                const std::string msg = json_value(data, "reasoning_budget_message", std::string());
                params.sampling.reasoning_budget_tokens = rb;
                params.sampling.reasoning_budget_start  = common_tokenize(vocab, think_start, false, true);
                params.sampling.reasoning_budget_end    = common_tokenize(vocab, think_end,   false, true);
                params.sampling.reasoning_budget_forced = common_tokenize(vocab, msg + think_end, false, true);
            } else {
                params.sampling.reasoning_budget_tokens = -1;
                params.sampling.reasoning_budget_start.clear();
                params.sampling.reasoning_budget_end.clear();
                params.sampling.reasoning_budget_forced.clear();
            }
        } else {
            params.sampling.reasoning_budget_tokens = defaults.sampling.reasoning_budget_tokens;
            params.sampling.reasoning_budget_start  = defaults.sampling.reasoning_budget_start;
            params.sampling.reasoning_budget_end    = defaults.sampling.reasoning_budget_end;
            params.sampling.reasoning_budget_forced = defaults.sampling.reasoning_budget_forced;
        }
    }

    params.speculative.n_min = json_value(data, "speculative.n_min", defaults.speculative.n_min);
    params.speculative.n_max = json_value(data, "speculative.n_max", defaults.speculative.n_max);
    params.speculative.p_min = json_value(data, "speculative.p_min", defaults.speculative.p_min);

    params.speculative.n_min = std::min(params.speculative.n_max, params.speculative.n_min);
    params.speculative.n_min = std::max(params.speculative.n_min, 0);
    params.speculative.n_max = std::max(params.speculative.n_max, 0);

    // Use OpenAI API logprobs only if n_probs wasn't provided
    if (data.contains("logprobs") && params.sampling.n_probs == defaults.sampling.n_probs){
        params.sampling.n_probs = json_value(data, "logprobs", defaults.sampling.n_probs);
    }

    if (data.contains("lora")) {
        if (data.at("lora").is_array()) {
            params.lora = parse_lora_request(params_base.lora_adapters, data.at("lora"));
        } else {
            throw std::runtime_error("Error: 'lora' must be an array of objects with 'id' and 'scale' fields");
        }
    } else {
        params.lora = params_base.lora_adapters;
    }

    // TODO: add more sanity checks for the input parameters

    if (params.sampling.penalty_last_n < -1) {
        throw std::runtime_error("Error: repeat_last_n must be >= -1");
    }

    if (params.sampling.dry_penalty_last_n < -1) {
        throw std::runtime_error("Error: dry_penalty_last_n must be >= -1");
    }

    if (params.sampling.penalty_last_n == -1) {
        // note: should be the worker's context and not the full context, but it's ok
        params.sampling.penalty_last_n = llama_n_ctx(ctx);
    }

    if (params.sampling.dry_penalty_last_n == -1) {
        params.sampling.dry_penalty_last_n = llama_n_ctx(ctx);
    }

    if (params.sampling.dry_base < 1.0f) {
        params.sampling.dry_base = defaults.sampling.dry_base;
    }

    // sequence breakers for DRY
    {
        // Currently, this is not compatible with TextGen WebUI, Koboldcpp and SillyTavern format
        // Ref: https://github.com/oobabooga/text-generation-webui/blob/d1af7a41ade7bd3c3a463bfa640725edb818ebaf/extensions/openai/typing.py#L39

        if (data.contains("dry_sequence_breakers")) {
            params.sampling.dry_sequence_breakers = json_value(data, "dry_sequence_breakers", std::vector<std::string>());
            if (params.sampling.dry_sequence_breakers.empty()) {
                throw std::runtime_error("Error: dry_sequence_breakers must be a non-empty array of strings");
            }
        }
    }

    // process "json_schema" and "grammar"
    if (data.contains("json_schema") && !data.contains("grammar")) {
        try {
            auto schema                  = json_value(data, "json_schema", json::object());
            SRV_DBG("JSON schema: %s\n", schema.dump(2).c_str());
            params.sampling.grammar      = json_schema_to_grammar(schema);
            SRV_DBG("Converted grammar: %s\n", params.sampling.grammar.c_str());
        } catch (const std::exception & e) {
            throw std::runtime_error(std::string("\"json_schema\": ") + e.what());
        }
    } else {
        params.sampling.grammar      = json_value(data, "grammar", defaults.sampling.grammar);
        SRV_DBG("Grammar: %s\n", params.sampling.grammar.c_str());
        params.sampling.grammar_lazy = json_value(data, "grammar_lazy", defaults.sampling.grammar_lazy);
        SRV_DBG("Grammar lazy: %s\n", params.sampling.grammar_lazy ? "true" : "false");
    }

    {
        auto it = data.find("chat_format");
        if (it != data.end()) {
            params.oaicompat_chat_format = static_cast<common_chat_format>(it->get<int>());
            SRV_INF("Chat format: %s\n", common_chat_format_name(params.oaicompat_chat_format).c_str());
        } else {
            params.oaicompat_chat_format = defaults.oaicompat_chat_format;
        }
    }

    {
        const auto preserved_tokens = data.find("preserved_tokens");
        if (preserved_tokens != data.end()) {
            for (const auto & t : *preserved_tokens) {
                auto ids = common_tokenize(vocab, t.get<std::string>(), /* add_special= */ false, /* parse_special= */ true);
                if (ids.size() == 1) {
                    SRV_DBG("Preserved token: %d\n", ids[0]);
                    params.sampling.preserved_tokens.insert(ids[0]);
                } else {
                    // This may happen when using a tool call style meant for a model with special tokens to preserve on a model without said tokens.
                    SRV_DBG("Not preserved because more than 1 token: %s\n", t.get<std::string>().c_str());
                }
            }
        }
        const auto grammar_triggers = data.find("grammar_triggers");
        if (grammar_triggers != data.end()) {
            for (const auto & t : *grammar_triggers) {
                auto ct = common_grammar_trigger::from_json(t);
                if (ct.type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
                    const auto & word = ct.value;
                    auto ids = common_tokenize(vocab, word, /* add_special= */ false, /* parse_special= */ true);
                    if (ids.size() == 1) {
                        auto token = ids[0];
                        if (std::find(params.sampling.preserved_tokens.begin(), params.sampling.preserved_tokens.end(), (llama_token) token) == params.sampling.preserved_tokens.end()) {
                            throw std::runtime_error("Grammar trigger word should be marked as preserved token: " + word);
                        }
                        SRV_DBG("Grammar trigger token: %d (`%s`)\n", token, word.c_str());
                        common_grammar_trigger trigger;
                        trigger.type = COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN;
                        trigger.value = word;
                        trigger.token = token;
                        params.sampling.grammar_triggers.push_back(std::move(trigger));
                    } else {
                        SRV_DBG("Grammar trigger word: `%s`\n", word.c_str());
                        params.sampling.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, word});
                    }
                } else {
                    params.sampling.grammar_triggers.push_back(ct);
                }
            }
        }
        if (params.sampling.grammar_lazy && params.sampling.grammar_triggers.empty()) {
            throw std::runtime_error("Error: no triggers set for lazy grammar!");
        }
    }

    {
        params.sampling.logit_bias.clear();
        params.ignore_eos = json_value(data, "ignore_eos", false);

        const auto & logit_bias = data.find("logit_bias");
        if (logit_bias != data.end() && logit_bias->is_array()) {
            const int n_vocab = llama_vocab_n_tokens(vocab);
            for (const auto & el : *logit_bias) {
                // TODO: we may want to throw errors here, in case "el" is incorrect
                if (el.is_array() && el.size() == 2) {
                    float bias;
                    if (el[1].is_number()) {
                        bias = el[1].get<float>();
                    } else if (el[1].is_boolean() && !el[1].get<bool>()) {
                        bias = -INFINITY;
                    } else {
                        continue;
                    }

                    if (el[0].is_number_integer()) {
                        llama_token tok = el[0].get<llama_token>();
                        if (tok >= 0 && tok < n_vocab) {
                            params.sampling.logit_bias.push_back({tok, bias});
                        }
                    } else if (el[0].is_string()) {
                        auto toks = common_tokenize(vocab, el[0].get<std::string>(), false);
                        for (auto tok : toks) {
                            params.sampling.logit_bias.push_back({tok, bias});
                        }
                    }
                }
            }
        }
    }

    {
        params.antiprompt.clear();

        const auto & stop = data.find("stop");
        if (stop != data.end() && stop->is_array()) {
            for (const auto & word : *stop) {
                if (!word.empty()) {
                    params.antiprompt.push_back(word);
                }
            }
        }
    }

    {
        const auto samplers = data.find("samplers");
        if (samplers != data.end()) {
            if (samplers->is_array()) {
                params.sampling.samplers = common_sampler_types_from_names(*samplers, false);
            } else if (samplers->is_string()){
                params.sampling.samplers = common_sampler_types_from_chars(samplers->get<std::string>());
            }
        } else {
            params.sampling.samplers = defaults.sampling.samplers;
        }
    }

    std::string model_name = params_base.model_alias.empty() ? DEFAULT_OAICOMPAT_MODEL : params_base.model_alias;
    params.oaicompat_model = json_value(data, "model", model_name);

    return params;
}

static std::unordered_set<int> get_list_id(const std::vector<ServerTask> & tasks) {
    std::unordered_set<int> ids(tasks.size());
    for (size_t i = 0; i < tasks.size(); i++) {
        ids.insert(tasks[i].id);
    }
    return ids;
}

json ServerTaskResultCmplFinal::to_json_non_oaicompat() {
    json res = json {
        {"index",               index},
        {"content",             stream ? "" : content}, // in stream mode, content is already in last partial chunk
        {"tokens",              stream ? llama_tokens {} : tokens},
        {"id_worker",             id_worker},
        {"stop",                true},
        {"model",               oaicompat_model},
        {"tokens_predicted",    n_decoded},
        {"tokens_evaluated",    n_prompt_tokens},
        {"generation_settings", generation_params.to_json()},
        {"prompt",              prompt},
        {"has_new_line",        has_new_line},
        {"truncated",           truncated},
        {"stop_type",           stop_type_to_str(stop)},
        {"stopping_word",       stopping_word},
        {"tokens_cached",       n_tokens_cached},
        {"timings",             timings.to_json()},
    };
    if (!stream && !probs_output.empty()) {
        res["completion_probabilities"] = CompletionTokenOutput::probs_vector_to_json(probs_output, post_sampling_probs);
    }
    return response_fields.empty() ? res : json_get_nested_values(response_fields, res);
}

json ServerTaskResultCmplFinal::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (!stream && probs_output.size() > 0) {
        logprobs = json{
            {"content", CompletionTokenOutput::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }
    json finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = "stop";
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          stream ? "" : content}, // in stream mode, content is already in last partial chunk
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", finish_reason},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", build_info},
        {"object",             "text_completion"},
        {"usage", json {
            {"completion_tokens", n_decoded},
            {"prompt_tokens",     n_prompt_tokens},
            {"total_tokens",      n_decoded + n_prompt_tokens}
        }},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (timings.prompt_n >= 0) {
        res.push_back({"timings", timings.to_json()});
    }

    return res;
}

json ServerTaskResultCmplFinal::to_json_oaicompat_chat() {
    std::string finish_reason = "length";
    common_chat_msg msg;
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        SRV_DBG("Parsing chat message: %s\n", content.c_str());
        msg = common_chat_parse(content, oaicompat_chat_format);
        finish_reason = msg.tool_calls.empty() ? "stop" : "tool_calls";
    } else {
        msg.content = content;
    }

    json message {
        {"role", "assistant"},
    };
    if (!msg.reasoning_content.empty()) {
        message["reasoning_content"] = msg.reasoning_content;
    }
    if (msg.content.empty() && !msg.tool_calls.empty()) {
        message["content"] = json();
    } else {
        message["content"] = msg.content;
    }
    if (!msg.tool_calls.empty()) {
        auto tool_calls = json::array();
        for (const auto & tc : msg.tool_calls) {
            tool_calls.push_back({
                {"type", "function"},
                {"function", {
                    {"name", tc.name},
                    {"arguments", tc.arguments},
                }},
                // Some templates generate and require an id (sometimes in a very specific format, e.g. Mistral Nemo).
                // We only generate a random id for the ones that don't generate one by themselves
                // (they also won't get to see it as their template likely doesn't use it, so it's all for the client)
                {"id", tc.id.empty() ? gen_tool_call_id() : tc.id},
            });
        }
        message["tool_calls"] = tool_calls;
    }

    json choice {
        {"finish_reason", finish_reason},
        {"index", 0},
        {"message", message},
    };

    if (!stream && probs_output.size() > 0) {
        choice["logprobs"] = json{
            {"content", CompletionTokenOutput::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }

    std::time_t t = std::time(0);

    json res = json {
        {"choices",            json::array({choice})},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", build_info},
        {"object",             "chat.completion"},
        {"usage", json {
            {"completion_tokens", n_decoded},
            {"prompt_tokens",     n_prompt_tokens},
            {"total_tokens",      n_decoded + n_prompt_tokens}
        }},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (timings.prompt_n >= 0) {
        res.push_back({"timings", timings.to_json()});
    }

    return res;
}

json ServerTaskResultCmplFinal::to_json_oaicompat_chat_stream() {
    std::time_t t = std::time(0);
    std::string finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = "stop";
    }

    json choice = json {
        {"finish_reason", finish_reason},
        {"index", 0},
        {"delta", json::object()}
    };

    json ret = json {
        {"choices",            json::array({choice})},
        {"created",            t},
        {"id",                 oaicompat_cmpl_id},
        {"model",              oaicompat_model},
        {"system_fingerprint", build_info},
        {"object",             "chat.completion.chunk"},
        {"usage", json {
            {"completion_tokens", n_decoded},
            {"prompt_tokens",     n_prompt_tokens},
            {"total_tokens",      n_decoded + n_prompt_tokens},
        }},
    };

    if (timings.prompt_n >= 0) {
        ret.push_back({"timings", timings.to_json()});
    }

    return ret;
}

json ServerTaskResultCmplPartial::to_json_non_oaicompat() {
    // non-OAI-compat JSON
    json res = json {
        {"index",            index},
        {"content",          content},
        {"tokens",           tokens},
        {"stop",             false},
        {"id_worker",          id_worker},
        {"tokens_predicted", n_decoded},
        {"tokens_evaluated", n_prompt_tokens},
    };
    // populate the timings object when needed (usually for the last response or with timings_per_token enabled)
    if (timings.prompt_n > 0) {
        res.push_back({"timings", timings.to_json()});
    }
    if (!prob_output.probs.empty()) {
        res["completion_probabilities"] = CompletionTokenOutput::probs_vector_to_json({prob_output}, post_sampling_probs);
    }
    return res;
}

json ServerTaskResultCmplPartial::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (prob_output.probs.size() > 0) {
        logprobs = json{
            {"content", CompletionTokenOutput::probs_vector_to_json({prob_output}, post_sampling_probs)},
        };
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", nullptr},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", build_info},
        {"object",             "text_completion"},
        {"id",                 oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (timings.prompt_n >= 0) {
        res.push_back({"timings", timings.to_json()});
    }

    return res;
}

json ServerTaskResultCmplPartial::to_json_oaicompat_chat() {
    bool first = n_decoded == 0;
    std::time_t t = std::time(0);
    json choices;

    if (first) {
        if (content.empty()) {
            choices = json::array({json{{"finish_reason", nullptr},
                                        {"index", 0},
                                        {"delta", json{{"role", "assistant"}}}}});
        } else {
            // We have to send this as two updates to conform to openai behavior
            json initial_ret = json{{"choices", json::array({json{
                                    {"finish_reason", nullptr},
                                    {"index", 0},
                                    {"delta", json{
                                        {"role", "assistant"}
                                    }}}})},
                        {"created", t},
                        {"id", oaicompat_cmpl_id},
                        {"model", oaicompat_model},
                        {"object", "chat.completion.chunk"}};

            json second_ret = json{
                        {"choices", json::array({json{{"finish_reason", nullptr},
                                                        {"index", 0},
                                                        {"delta", json {
                                                        {"content", content}}}
                                                        }})},
                        {"created", t},
                        {"id", oaicompat_cmpl_id},
                        {"model", oaicompat_model},
                        {"object", "chat.completion.chunk"}};

            return std::vector<json>({initial_ret, second_ret});
        }
    } else {
        choices = json::array({json{
            {"finish_reason", nullptr},
            {"index", 0},
            {"delta",
            json {
                {"content", content},
            }},
        }});
    }

    GGML_ASSERT(choices.size() >= 1);

    if (prob_output.probs.size() > 0) {
        choices[0]["logprobs"] = json{
            {"content", CompletionTokenOutput::probs_vector_to_json({prob_output}, post_sampling_probs)},
        };
    }

    json ret = json {
        {"choices",            choices},
        {"created",            t},
        {"id",                 oaicompat_cmpl_id},
        {"model",              oaicompat_model},
        {"system_fingerprint", build_info},
        {"object",             "chat.completion.chunk"}
    };

    if (timings.prompt_n >= 0) {
        ret.push_back({"timings", timings.to_json()});
    }

    return std::vector<json>({ret});
}

bool ServerTaskResultError::is_error() {
    return true;
}

json ServerTaskResultError::to_json() {
    return format_error_response(err_msg, err_type);
}

}