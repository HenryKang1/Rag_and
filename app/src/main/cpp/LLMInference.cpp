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
    return std::string(result.data(), result.size());
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
    return (float)_responseNumTokens / (_responseGenerationTime / 1e6);
}

int
LLMInference::getContextSizeUsed() const {
    return _nCtxUsed;
}

void
LLMInference::startCompletion(const char* query) {
    if (!_storeChats) {
        _prevLen = 0;
        _formattedMessages.clear();
        _formattedMessages = std::vector<char>(llama_n_ctx(_ctx));
        _messages.clear();
        // Clear KV cache for fresh start
        llama_memory_clear(llama_get_memory(_ctx), 0);
    }
    _responseGenerationTime = 0;
    _responseNumTokens      = 0;
    addChatMessage(query, "user");
    // apply the chat-template
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
    std::string prompt(_formattedMessages.begin() + _prevLen, _formattedMessages.begin() + newLen);
    _promptTokens = common_tokenize(llama_model_get_vocab(_model), prompt, true, true);

    // create a llama_batch containing a single sequence
    // Use SmolChat-Android's simple approach
    _batch.token    = _promptTokens.data();
    _batch.n_tokens = _promptTokens.size();
    
    LOGi("[DEBUG] Batch prepared with %d prompt tokens", _batch.n_tokens);
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
        LOGi("  Batch tokens: %d", _batch.n_tokens);
    }
    
    if (_nCtxUsed + _batch.n_tokens > contextSize) {
        LOGe("Context size reached! Used: %d, Batch: %d, Total: %d > %d",
             _nCtxUsed, _batch.n_tokens, 
             _nCtxUsed + _batch.n_tokens, contextSize);
        throw std::runtime_error("context size reached");
    }

    auto start = ggml_time_us();
    // run the model
    LOGi("[DEBUG] Calling llama_decode with batch.n_tokens=%d", _batch.n_tokens);
    int decode_result = llama_decode(_ctx, _batch);
    if (decode_result < 0) {
        LOGe("[DEBUG] llama_decode failed with result=%d", decode_result);
        LOGe("[DEBUG] Current state: _responseNumTokens=%ld, batch.pos[0]=%d", _responseNumTokens, _batch.pos[0]);
        throw std::runtime_error("llama_decode() failed");
    }
    LOGi("[DEBUG] llama_decode succeeded");

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
    _responseNumTokens += 1;
    _cacheResponseTokens += piece;

    // re-init the batch with the newly predicted token
    // key, value pairs of all previous tokens have been cached in the KV cache
    // Use SmolChat-Android's simple approach
    _batch.token    = &_currToken;
    _batch.n_tokens = 1;

    if (_isValidUtf8(_cacheResponseTokens.c_str())) {
        _response += _cacheResponseTokens;
        std::string valid_utf8_piece = _cacheResponseTokens;
        _cacheResponseTokens.clear();
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
    
    // Clear the batch and context usage for next generation
    _batch.n_tokens = 0;
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
    
    // Clear batch
    if (_batch.token) {
        _batch.n_tokens = 0;
    }
    
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
    // Free the batch structure
    llama_batch_free(_batch);
    if (_sampler) {
        llama_sampler_free(_sampler);
    }
    llama_free(_ctx);
    llama_model_free(_model);
}