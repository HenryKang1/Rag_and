#include "LLMInference.h"
#include "llama.h"
#include "gguf.h"
#include <android/log.h>
#include <cstring>
#include <iostream>

#define TAG "LlamaCpp-Native"
#define LOGi(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGe(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// Helper function implementations for tokenization
static std::vector<llama_token> common_tokenize(
    const struct llama_vocab * vocab,
    const std::string & text,
    bool   add_special,
    bool   parse_special = false) {
    // Tokenize text using llama's tokenizer
    std::vector<llama_token> result;
    const int n_max_tokens = text.length() + (add_special ? 2 : 0);
    result.resize(n_max_tokens);
    const int n = llama_tokenize(vocab, text.c_str(), text.length(), result.data(), n_max_tokens, add_special, parse_special);
    if (n < 0) {
        result.resize(-n);
        llama_tokenize(vocab, text.c_str(), text.length(), result.data(), result.size(), add_special, parse_special);
    } else {
        result.resize(n);
    }
    return result;
}

static std::string common_token_to_piece(
    const struct llama_context * ctx,
    llama_token   token,
    bool          special = true) {
    std::vector<char> result(8, 0);
    const int n_tokens = llama_token_to_piece(llama_model_get_vocab(llama_get_model(ctx)), token, result.data(), result.size(), 0, special);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_token_to_piece(llama_model_get_vocab(llama_get_model(ctx)), token, result.data(), result.size(), 0, special);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }
    return std::s57tring(result.data(), result.size());
}

void
LLMInference::loadModel(const char* model_path, float minP, float temperature, bool storeChats, long contextSize,
                        const char* chatTemplate, int nThreads, bool useMmap, bool useMlock, float topP, int topK,
                        float xtcP, float xtcT) {
    LOGi("loading model with"
         "\n\tmodel_path = %s"
         "\n\tminP = %f"
         "\n\ttemperature = %f"
         "\n\tstoreChats = %d"
         "\n\tcontextSize = %d"
         "\n\tchatTemplate = %s"
         "\n\tnThreads = %d"
         "\n\tuseMmap = %d"
         "\n\tuseMlock = %d"
         "\n\ttopP = %f"
         "\n\ttopK = %i"
         "\n\txtcP = %f"
         "\n\txtcT = %f",
         model_path, minP, temperature, (int)storeChats, (int)contextSize, chatTemplate, nThreads, useMmap, useMlock, topP, topK, xtcP, xtcT);

    // create an instance of llama_model
    llama_model_params model_params = llama_model_default_params();
    model_params.use_mmap           = useMmap;
    model_params.use_mlock          = useMlock;
    _model                          = llama_model_load_from_file(model_path, model_params);

    if (!_model) {
        LOGe("failed to load model from %s", model_path);
        throw std::runtime_error("loadModel() failed");
    }

    // create an instance of llama_context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx                = contextSize;
    ctx_params.n_threads            = nThreads;
    ctx_params.no_perf              = true; // disable performance metrics
    // n_batch 설정: 프리필 성능 최적화를 위해 적절한 크기 설정
    ctx_params.n_batch              = std::min<int>(1024, (int)contextSize);
    _ctx                            = llama_init_from_model(_model, ctx_params);

    if (!_ctx) {
        LOGe("llama_new_context_with_model() returned null)");
        throw std::runtime_error("llama_new_context_with_model() returned null");
    }
    
    // Log actual context size
    LOGi("Context created successfully:");
    LOGi("  Requested context size: %d", (int)contextSize);
    LOGi("  Actual context size: %d", llama_n_ctx(_ctx));
    LOGi("  Batch size: %d", llama_n_batch(_ctx));
    LOGi("  Model vocab size: %d", llama_vocab_n_tokens(llama_model_get_vocab(_model)));

    // initialize sampler with safer settings
    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    sampler_params.no_perf                    = true; // disable performance metrics
    _sampler                                  = llama_sampler_chain_init(sampler_params);

    // Add samplers in safer order - filtering first, then temperature
    if (topK > 0 && topK <= 100) {  // Limit top-k to reasonable range
        LOGi("Enabled top-k sampling with k=%d", topK);
        llama_sampler_chain_add(_sampler, llama_sampler_init_top_k(topK));
    }
    if (topP > 0.0f && topP <= 0.99f) {
        LOGi("Enabled top-p sampling with p=%f", topP);
        llama_sampler_chain_add(_sampler, llama_sampler_init_top_p(topP, 1));
    }
    if (minP >= 0.01f && minP < 1.0f) {
        LOGi("Enabled min-p sampling with p=%f", minP);
        llama_sampler_chain_add(_sampler, llama_sampler_init_min_p(minP, 1));
    }
    // Disable XTC sampling as it might cause instability
    // if (xtcT <= 0.99 || xtcP >= 0.01) {
    //     LOGi("Enabled XTC sampling with p=%f, t=%f", xtcP, xtcT);
    //     llama_sampler_chain_add(_sampler, llama_sampler_init_xtc(xtcP, xtcT, 1, LLAMA_DEFAULT_SEED));
    // }
    
    // Temperature and dist should be last
    llama_sampler_chain_add(_sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(_sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    _formattedMessages = std::vector<char>(llama_n_ctx(_ctx));
    _messages.clear();
    // If chatTemplate is null or empty, use nullptr for default template
    if (chatTemplate && strlen(chatTemplate) > 0) {
        _chatTemplate = strdup(chatTemplate);
    } else {
        _chatTemplate = nullptr;
    }
    this->_storeChats = storeChats;
    
    // Don't initialize batch with llama_batch_init like SmolChat-Android
    // The batch will be set up in startCompletion
}

void
LLMInference::addChatMessage(const char* message, const char* role) {
    _messages.push_back({ strdup(role), strdup(message) });
}

float
LLMInference::getResponseGenerationTime() const {
    if (_responseGenerationTime <= 0) return 0.0f;
    return (float)_responseNumTokens / (_responseGenerationTime / 1e6);
}

int
LLMInference::getContextSizeUsed() const {
    return _nCtxUsed;
}

void
LLMInference::startCompletion(const char* query) {
    // Reset metrics
    _responseGenerationTime = 0;
    _responseNumTokens      = 0;
    _prefillTimeUs = 0;
    _decodeTimeUs = 0;
    _tPrefillStartUs = 0;
    _firstTokenLatencyUs = 0;
    _firstTokenLogged = false;
    _currPos = 0;  // 위치 초기화
    
    // Check if prompt already contains chat template tags (from Kotlin buildRAGPrompt)
    const std::string q(query ? query : "");
    const bool has_chat_tags = q.find("<|im_start|>") != std::string::npos || 
                               q.find("<|im_end|>") != std::string::npos;
    
    if (!_storeChats) {
        _prevLen = 0;
        _formattedMessages.clear();
        _formattedMessages = std::vector<char>(llama_n_ctx(_ctx));
        _messages.clear();
        // Clear KV cache for fresh start
        llama_memory_clear(llama_get_memory(_ctx), 0);
    }
    
    std::string prompt;
    if (has_chat_tags) {
        // Already formatted prompt from Kotlin (buildRAGPrompt), use as-is
        prompt = q;
        LOGi("[PROMPT] Using pre-formatted prompt with chat template tags (length: %zu)", prompt.length());
    } else {
        // Apply chat template to raw user message
        addChatMessage(query, "user");
        int newLen = llama_chat_apply_template(_chatTemplate, _messages.data(), _messages.size(), true,
                                               _formattedMessages.data(), _formattedMessages.size());
        if (newLen > (int)_formattedMessages.size()) {
            // resize the output buffer `_formattedMessages`
            // and re-apply the chat template
            _formattedMessages.resize(newLen);
            newLen = llama_chat_apply_template(_chatTemplate, _messages.data(), _messages.size(), true,
                                               _formattedMessages.data(), _formattedMessages.size());
        }
        if (newLen < 0) {
            throw std::runtime_error("llama_chat_apply_template() in LLMInference::startCompletion() failed");
        }
        prompt.assign(_formattedMessages.begin() + _prevLen, _formattedMessages.begin() + newLen);
        LOGi("[PROMPT] Applied chat template to user message (length: %zu)", prompt.length());
    }
    
    _promptTokens = common_tokenize(llama_model_get_vocab(_model), prompt, true, true);

    // 프리필을 n_batch 크기로 청크 나눠 처리 (성능 최적화)
    const int32_t n_batch = llama_n_batch(_ctx);
    const int n_prompt_tokens = static_cast<int>(_promptTokens.size());
    
    _tPrefillStartUs = ggml_time_us();
    _prefillTimeUs = 0;
    LOGi("[PERF] PREFILL start: prompt_tokens=%d, n_batch=%d", n_prompt_tokens, n_batch);
    
    // 프리필이 n_batch보다 작거나 같으면 한 번에 처리
    if (n_prompt_tokens <= n_batch) {
        // Properly initialize batch instead of using llama_batch_get_one
        llama_batch batch = llama_batch_init(n_prompt_tokens, 0, 1);
        for (int i = 0; i < n_prompt_tokens; ++i) {
            batch.token[i] = _promptTokens[i];
            batch.pos[i] = i;
            batch.seq_id[i][0] = 0;
            batch.n_seq_id[i] = 1;
            batch.logits[i] = (i == n_prompt_tokens - 1); // Only last token needs logits
        }
        batch.n_tokens = n_prompt_tokens;
        
        auto chunk_start = ggml_time_us();
        int decode_result = llama_decode(_ctx, batch);
        auto chunk_end = ggml_time_us();
        
        if (decode_result < 0) {
            LOGe("[PERF] PREFILL failed, result=%d", decode_result);
            llama_batch_free(batch);
            throw std::runtime_error("llama_decode() failed during prefill");
        }
        
        _prefillTimeUs = chunk_end - chunk_start;
        llama_batch_free(batch);
    } else {
        // 큰 프리필을 청크로 나눠 처리 (llama_batch_init으로 재사용 가능한 배치 생성)
        llama_batch batch = llama_batch_init(n_batch, 0, 1);
        int consumed = 0;
        
        while (consumed < n_prompt_tokens) {
            batch.n_tokens = 0;  // 배치 클리어
            
            const int take = std::min(n_batch, n_prompt_tokens - consumed);
            for (int i = 0; i < take; ++i) {
                const int idx = consumed + i;
                const llama_token tok = _promptTokens[idx];
                const bool need_logits = (idx == n_prompt_tokens - 1); // 마지막 프리필 토큰만 logits
                
                // 배치에 토큰 추가 (명시적 위치와 logits 설정)
                batch.token[batch.n_tokens] = tok;
                batch.pos[batch.n_tokens] = (llama_pos)idx;
                batch.n_seq_id[batch.n_tokens] = 1;
                batch.seq_id[batch.n_tokens][0] = 0;
                batch.logits[batch.n_tokens] = need_logits;
                batch.n_tokens++;
            }
            
            auto chunk_start = ggml_time_us();
            int decode_result = llama_decode(_ctx, batch);
            auto chunk_end = ggml_time_us();
            
            if (decode_result < 0) {
                LOGe("[PERF] PREFILL failed at chunk consumed=%d, result=%d", consumed, decode_result);
                llama_batch_free(batch);
                throw std::runtime_error("llama_decode() failed during prefill");
            }
            
            _prefillTimeUs += (chunk_end - chunk_start);
            if (consumed == 0) {
                LOGi("[PERF] PREFILL first chunk: %d tokens in %.2f ms", 
                     take, (chunk_end - chunk_start) / 1000.0f);
            }
            
            consumed += take;
        }
        
        llama_batch_free(batch);
    }
    
    // 프리필 완료 후 디코드 위치 초기화
    _currPos = (llama_pos)n_prompt_tokens;
    
    auto prefill_end = ggml_time_us();
    const double prefill_ms = _prefillTimeUs / 1000.0;
    const double prefill_wall_ms = (prefill_end - _tPrefillStartUs) / 1000.0;
    LOGi("[PERF] PREFILL done: %d tokens | wall %.2f ms | kernel %.2f ms | %.2f tok/s", 
         n_prompt_tokens,
         prefill_wall_ms,
         prefill_ms,
         n_prompt_tokens > 0 ? (1000.0 * (double)n_prompt_tokens / prefill_wall_ms) : 0.0);
}

// taken from:
// https://github.com/ggerganov/llama.cpp/blob/master/examples/llama.android/llama/src/main/cpp/llama-android.cpp#L38
bool
LLMInference::_isValidUtf8(const char* response) {
    if (!response) {
        return true;
    }
    const unsigned char* bytes = (const unsigned char*)response;
    int                  num;
    while (*bytes != 0x00) {
        if ((*bytes & 0x80) == 0x00) {
            // U+0000 to U+007F
            num = 1;
        } else if ((*bytes & 0xE0) == 0xC0) {
            // U+0080 to U+07FF
            num = 2;
        } else if ((*bytes & 0xF0) == 0xE0) {
            // U+0800 to U+FFFF
            num = 3;
        } else if ((*bytes & 0xF8) == 0xF0) {
            // U+10000 to U+10FFFF
            num = 4;
        } else {
            return false;
        }

        bytes += 1;
        for (int i = 1; i < num; ++i) {
            if ((*bytes & 0xC0) != 0x80) {
                return false;
            }
            bytes += 1;
        }
    }
    return true;
}

std::string
LLMInference::completionLoop() {
    // check if the length of the inputs to the model
    // have exceeded the context size of the model
    uint32_t contextSize = llama_n_ctx(_ctx);
    // Approximate context usage since llama_kv_self_used_cells is not available
    _nCtxUsed = _promptTokens.size() + _responseNumTokens;
    
    // Log every 10 tokens or when getting close to limit
    if (_responseNumTokens % 10 == 0 || _nCtxUsed > contextSize - 100) {
        LOGi("Token generation status:");
        LOGi("  Generated tokens: %d", _responseNumTokens);
        LOGi("  Context used: %d / %d (%.1f%%)", 
             _nCtxUsed, contextSize, 
             (_nCtxUsed * 100.0f / contextSize));
    }
    
    // Context 체크 (다음 토큰 포함)
    if (_nCtxUsed + 1 > contextSize) {
        LOGe("Context size reached! Used: %d, would exceed: %d",
             _nCtxUsed, contextSize);
        throw std::runtime_error("context size reached");
    }

    auto start = ggml_time_us();
    
    // 첫 토큰: 프리필에서 이미 logits 계산됨, decode 불필요, 바로 샘플링
    // 이후 토큰들: 이전에 샘플된 토큰을 decode하고 새로 샘플링
    if (_responseNumTokens > 0) {
        // 이전 루프에서 샘플된 토큰을 디코드 (명시적 위치 사용)
        llama_batch dec_batch = llama_batch_init(1, 0, 1);
        dec_batch.token[0] = _currToken;
        dec_batch.pos[0] = _currPos++;
        dec_batch.n_seq_id[0] = 1;
        dec_batch.seq_id[0][0] = 0;
        dec_batch.logits[0] = true;
        dec_batch.n_tokens = 1;
        
        LOGi("[DEBUG] Calling llama_decode with batch.n_tokens=%d (decode step, pos=%d)", 
             dec_batch.n_tokens, (int)dec_batch.pos[0]);
        
        int decode_result = llama_decode(_ctx, dec_batch);
        llama_batch_free(dec_batch);
        
        if (decode_result < 0) {
            LOGe("[DEBUG] llama_decode failed with result=%d", decode_result);
            LOGe("[DEBUG] Current state: _responseNumTokens=%ld, _currPos=%d", 
                 _responseNumTokens, (int)_currPos);
            throw std::runtime_error("llama_decode() failed");
        }
        _decodeTimeUs += (ggml_time_us() - start);
    } else {
        // 첫 토큰: 프리필에서 이미 logits 계산됨, decode 스킵
        LOGi("[DEBUG] First token: using logits from prefill, skipping decode");
    }

    // sample a token and check if it is an EOG (end of generation token)
    // convert the integer token to its corresponding word-piece
    if (!_sampler) {
        LOGe("Sampler is null!");
        throw std::runtime_error("Sampler is null");
    }
    if (!_ctx) {
        LOGe("Context is null!");
        throw std::runtime_error("Context is null");
    }
    
    // Try to sample token with error handling
    try {
        _currToken = llama_sampler_sample(_sampler, _ctx, -1);
    } catch (const std::exception &e) {
        LOGe("Exception during sampling: %s", e.what());
        throw std::runtime_error("Sampling failed");
    }
    
    // Validate token
    const auto vocab = llama_model_get_vocab(_model);
    const int vocab_size = llama_vocab_n_tokens(vocab);
    
    if (_currToken < 0 || _currToken >= vocab_size) {
        LOGe("Invalid token sampled: %d (vocab size: %d)", _currToken, vocab_size);
        // Try to recover by returning end of generation
        return "[EOG]";
    }
    
    if (llama_vocab_is_eog(vocab, _currToken)) {
        addChatMessage(strdup(_response.data()), "assistant");
        _response.clear();
        return "[EOG]";
    }
    std::string piece = common_token_to_piece(_ctx, _currToken, true);
    LOGi("common_token_to_piece: %s", piece.c_str());
    auto end = ggml_time_us();
    _responseGenerationTime += (end - start);
    if (_responseNumTokens == 0 && !_firstTokenLogged) {
        // first token latency = time from prefill start to first token materialized
        _firstTokenLatencyUs = end - _tPrefillStartUs;
        _firstTokenLogged = true;
        // report initial perf snapshot
        LOGi("[PERF] FIRST TOKEN latency: %lld us (%.2f ms)",
             (long long)_firstTokenLatencyUs, _firstTokenLatencyUs / 1000.0f);
    }
    _responseNumTokens += 1;
    _cacheResponseTokens += piece;

    // 다음 스텝은 completionLoop에서 명시적 배치로 처리됨

    if (_isValidUtf8(_cacheResponseTokens.c_str())) {
        _response += _cacheResponseTokens;
        std::string valid_utf8_piece = _cacheResponseTokens;
        _cacheResponseTokens.clear();
        // periodic decode perf (exclude prefill)
        if (_responseNumTokens % 10 == 0) {
            // tokens after first token
            long decoded_tokens = (_responseNumTokens > 0) ? (_responseNumTokens - 1) : 0;
            double decode_ms = _decodeTimeUs / 1000.0;
            double tps = (decode_ms > 0.0 && decoded_tokens > 0)
                         ? (decoded_tokens * 1000.0 / decode_ms)
                         : 0.0;
            LOGi("[PERF] DECODE progress: tokens=%ld, decode_time=%.2f ms, speed=%.2f tok/s",
                 decoded_tokens, decode_ms, tps);
        }
        return valid_utf8_piece;
    }

    return "";
}

void
LLMInference::stopCompletion() {
    if (_storeChats) {
        addChatMessage(_response.c_str(), "assistant");
    }
    _response.clear();
    _cacheResponseTokens.clear();
    _responseNumTokens = 0;
    _responseGenerationTime = 0;
    
    // Clear context usage for next generation
    // Note: _batch is managed in completionLoop, no need to clear here
    _nCtxUsed = 0;
    
    // Clear KV cache for clean start
    if (_ctx) {
        llama_memory_t mem = llama_get_memory(_ctx);
        if (mem) {
            llama_memory_clear(mem, false);  // false = clear metadata only, not data buffers
        }
    }
    
    // Reset sampler state for next generation
    if (_sampler) {
        llama_sampler_reset(_sampler);
    }
    
    const char* tmpl = llama_model_chat_template(_model, nullptr);
    _prevLen         = llama_chat_apply_template(tmpl, _messages.data(), _messages.size(), false, nullptr, 0);
    if (_prevLen < 0) {
        throw std::runtime_error("llama_chat_apply_template() in LLMInference::stopCompletion() failed");
    }
    
    LOGi("Completion stopped. Context cleared.");
}

void LLMInference::clearContext() {
    LOGi("Clearing context and memory for single-turn QA");
    
    // Clear KV cache using llama_memory API
    if (_ctx) {
        llama_memory_t mem = llama_get_memory(_ctx);
        if (mem) {
            llama_memory_clear(mem, true);  // true = clear both metadata and data buffers
            LOGi("Memory cache cleared");
        }
    }
    
    // Clear message history
    _messages.clear();
    _formattedMessages.clear();
    _promptTokens.clear();
    _response.clear();
    _cacheResponseTokens.clear();
    
    // Reset context usage
    _nCtxUsed = 0;
    _prevLen = 0;
    _responseNumTokens = 0;
    _responseGenerationTime = 0;
    
    // Clear batch (will be recreated in startCompletion)
    // Note: _batch is freed after prefill, no need to clear here
    
    // Reset sampler
    if (_sampler) {
        llama_sampler_reset(_sampler);
    }
    
    LOGi("Context and memory cleared successfully");
}

LLMInference::~LLMInference() {
    LOGi("deallocating LLMInference instance");
    // free memory held by the message text in messages
    // (as we had used strdup() to create a malloc'ed copy)
    for (llama_chat_message& message : _messages) {
        free(const_cast<char*>(message.role));
        free(const_cast<char*>(message.content));
    }
    if (_chatTemplate) {
        free(const_cast<char*>(_chatTemplate));
    }
    // Note: _batch is always freed immediately after use in startCompletion,
    // so there's no need to free it here. Freeing an uninitialized batch would cause a crash.
    if (_sampler) {
        llama_sampler_free(_sampler);
    }
    llama_free(_ctx);
    llama_model_free(_model);
}