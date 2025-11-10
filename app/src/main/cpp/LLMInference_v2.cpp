// LLMInference.cpp  — mobile-friendly prompt decode + incremental decode
// (c) your project

#include "LLMInference.h"

#include "llama.h"
#include "gguf.h"

#include <android/log.h>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>

#define TAG "LlamaCpp-Native"
#define LOGi(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGe(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// ---------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------

// 간단 UTF-8 유효성 검사 (안전)
bool LLMInference::_isValidUtf8(const char* response) {
    if (!response) return true;

    const unsigned char* s = reinterpret_cast<const unsigned char*>(response);
    while (*s != 0x00) {
        unsigned char c = *s;
        int need = 0;

        if ((c & 0x80) == 0x00) {                    // 1-byte
            s += 1; continue;
        } else if ((c & 0xE0) == 0xC0) {             // 2-byte
            need = 1;
            if (c < 0xC2) return false;              // overlong
        } else if ((c & 0xF0) == 0xE0) {             // 3-byte
            need = 2;
        } else if ((c & 0xF8) == 0xF0) {             // 4-byte
            need = 3;
            if (c > 0xF4) return false;              // > U+10FFFF
        } else {
            return false;                             // invalid lead
        }

        s += 1;
        for (int i = 0; i < need; ++i) {
            unsigned char cc = s[i];
            if ((cc & 0xC0) != 0x80) return false;   // continuation must be 10xxxxxx
        }

        // extra checks
        if (need == 2) {
            unsigned char c1 = *(s - 1);
            unsigned char c2 = s[0];
            if (c1 == 0xE0 && (c2 & 0xE0) == 0x80) return false; // overlong 3-byte
            if (c1 == 0xED && (c2 & 0xE0) == 0xA0) return false; // surrogate
        } else if (need == 3) {
            unsigned char c1 = *(s - 1);
            unsigned char c2 = s[0];
            if (c1 == 0xF0 && (c2 & 0xF0) == 0x80) return false; // overlong 4-byte
            if (c1 == 0xF4 && (c2 & 0xF0) >  0x80) return false; // > U+10FFFF
        }

        s += need;
    }
    return true;
}

// ---------------------------------------------------------------
// Core methods
// ---------------------------------------------------------------

void
LLMInference::loadModel(const char* model_path,
                        float minP, float temperature, bool storeChats, long contextSize,
                        const char* chatTemplate, int nThreads, bool useMmap, bool useMlock,
                        float topP, int topK, float xtcP, float xtcT) {
    LOGi("loading model with"
         "\n\tmodel_path = %s"
         "\n\tminP = %f"
         "\n\ttemperature = %f"
         "\n\tstoreChats = %d"
         "\n\tcontextSize = %ld"
         "\n\tchatTemplate = %s"
         "\n\tnThreads = %d"
         "\n\tuseMmap = %d"
         "\n\tuseMlock = %d"
         "\n\ttopP = %f"
         "\n\ttopK = %d"
         "\n\txtcP = %f"
         "\n\txtcT = %f",
         model_path, minP, temperature, storeChats, contextSize,
         (chatTemplate ? chatTemplate : "(null)"),
         nThreads, useMmap, useMlock, topP, topK, xtcP, xtcT);

    // --- model params ---
    llama_model_params model_params = llama_model_default_params();
    model_params.use_mmap  = useMmap;
    model_params.use_mlock = useMlock;

    // 최신 C API: llama_model_load_from_file (deprecated 교체)
    _model = llama_model_load_from_file(model_path, model_params);
    if (!_model) {
        LOGe("llama_model_load_from_file() failed");
        throw std::runtime_error("llama_model_load_from_file() failed");
    }

    // --- context params ---
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx     = (int)contextSize;
    ctx_params.n_threads = nThreads;
    ctx_params.n_batch   = 512;      // 모바일 프롬프트 디코딩 가속(단말 메모리에 맞춰 256~1024 튜닝)
    ctx_params.no_perf   = true;

    _ctx = llama_init_from_model(_model, ctx_params);
    if (!_ctx) {
        LOGe("llama_init_from_model() returned null");
        throw std::runtime_error("llama_init_from_model() returned null");
    }

    LOGi("Context created:");
    LOGi("  n_ctx=%d  n_batch=%d  vocab=%d",
         llama_n_ctx(_ctx), 512, llama_vocab_n_tokens(llama_model_get_vocab(_model)));

    // --- sampler chain ---
    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    sampler_params.no_perf = true;
    _sampler = llama_sampler_chain_init(sampler_params);

    if (topK > 0)                   llama_sampler_chain_add(_sampler, llama_sampler_init_top_k(topK));
    if (topP > 0.0f && topP < 1.0f) llama_sampler_chain_add(_sampler, llama_sampler_init_top_p(topP, 1));
    if (minP > 0.0f && minP < 1.0f) llama_sampler_chain_add(_sampler, llama_sampler_init_min_p(minP, 1));
    // XTC는 4개 인자 (p, t, min_keep, seed)
    if (xtcP > 0.0f && xtcT > 0.0f) llama_sampler_chain_add(_sampler, llama_sampler_init_xtc(xtcP, xtcT, /*min_keep=*/1, /*seed=*/0));
    // temperature 적용(원치 않으면 아래 한 줄 주석 처리)
    // llama_sampler_chain_add(_sampler, llama_sampler_init_temp(temperature));

    // --- chat template & state ---
    if (chatTemplate && std::strlen(chatTemplate) > 0) {
        _chatTemplate = strdup(chatTemplate);
    } else {
        _chatTemplate = nullptr;
    }
    _storeChats = storeChats;

    _messages.clear();
    _formattedMessages.clear();
    _promptTokens.clear();
    _response.clear();
    _cacheResponseTokens.clear();
    _prevLen = 0;
    _nCtxUsed = 0;
    _responseNumTokens = 0;
    _responseGenerationTime = 0;

    // 안전한 파괴를 위해 최소 배치 초기화(로컬 배치를 주로 쓰지만 멤버도 보유)
    _batch = llama_batch_init(/*n_tokens_max=*/1, /*embd=*/0, /*n_seq_max=*/1);
}

void
LLMInference::startCompletion(const char* query) {
    // (1) 채팅 상태 초기화 & 템플릿 버퍼 준비
    if (!_storeChats) {
        _prevLen = 0;
        _formattedMessages.clear();
        _formattedMessages = std::vector<char>(llama_n_ctx(_ctx) * 8); // 문자 버퍼 넉넉히
        _messages.clear();
        // 새 턴 시작 시 KV 캐시를 "깊게" 1회 초기화
        llama_kv_cache_clear(_ctx);
        llama_kv_cache_clear_drafts(_ctx);
    }

    _response.clear();
    _cacheResponseTokens.clear();
    _responseGenerationTime = 0;
    _responseNumTokens = 0;

    // (2) 유저 메시지 추가 후 템플릿 적용 → 프롬프트 문자열 생성
    addChatMessage(query, "user");

    const char* tmpl = _chatTemplate ? _chatTemplate : llama_model_chat_template(_model, nullptr);

    int newLen = llama_chat_apply_template(tmpl,
                                           _messages.data(), _messages.size(),
                                           /*add_assistant=*/true,
                                           _formattedMessages.data(), _formattedMessages.size());
    if (newLen > (int)_formattedMessages.size()) {
        _formattedMessages.resize(newLen);
        newLen = llama_chat_apply_template(tmpl,
                                           _messages.data(), _messages.size(),
                                           /*add_assistant=*/true,
                                           _formattedMessages.data(), _formattedMessages.size());
    }
    if (newLen < 0) {
        throw std::runtime_error("llama_chat_apply_template() failed in startCompletion()");
    }

    std::string prompt(_formattedMessages.begin() + _prevLen,
                       _formattedMessages.begin() + newLen);

    // (토크나이즈) — C API 사용
    {
        _promptTokens.clear();
        const llama_vocab * vocab = llama_model_get_vocab(_model);
        // add_special=true, parse_special=true
        llama_tokenize(vocab,
                       prompt.c_str(),
                       /*add_special=*/true,
                       /*parse_special=*/true,
                       _promptTokens);
    }

    // (3) 프롬프트 배치를 명시적으로 구성하고 한 번만 디코드
    {
        const int n_prompt = (int)_promptTokens.size();
        llama_batch prompt_batch = llama_batch_init(/*n_tokens_max=*/n_prompt, /*embd=*/0, /*n_seq_max=*/1);

        for (int i = 0; i < n_prompt; ++i) {
            prompt_batch.token[i]  = _promptTokens[i];
            prompt_batch.pos[i]    = i;                   // 0..n_prompt-1
            prompt_batch.seq_id[i] = 0;                   // 단일 시퀀스
            prompt_batch.logits[i] = (i == n_prompt - 1); // 마지막 토큰만 로짓 계산
        }
        prompt_batch.n_tokens = n_prompt;

        LOGi("[PROMPT] llama_decode(prompt) n=%d", n_prompt);
        const int rc = llama_decode(_ctx, prompt_batch);
        llama_batch_free(prompt_batch);
        if (rc < 0) {
            throw std::runtime_error("llama_decode() failed on prompt");
        }

        _nCtxUsed = n_prompt;
    }

    _prevLen = newLen; // 다음 템플릿 슬라이싱 기준
    // 다음 completionLoop() 첫 호출에서는 "프롬프트 로짓"으로 바로 샘플 → 이후부터 1-스텝 증분
}

std::string
LLMInference::completionLoop() {
    const uint32_t contextSize = llama_n_ctx(_ctx);

    const auto t0 = ggml_time_us();
    if (_responseNumTokens > 0) {
        // 증분 1-토큰 디코드
        llama_batch step = llama_batch_init(/*n_tokens_max=*/1, /*embd=*/0, /*n_seq_max=*/1);
        step.token[0]  = _currToken;       // 직전 스텝에서 고른 토큰
        step.pos[0]    = _nCtxUsed;        // 누적 위치
        step.seq_id[0] = 0;
        step.logits[0] = true;             // 이번 스텝의 로짓 필요
        step.n_tokens  = 1;

        LOGi("[STEP] llama_decode(pos=%d, tok=%d)", step.pos[0], step.token[0]);
        const int rc = llama_decode(_ctx, step);
        llama_batch_free(step);
        if (rc < 0) {
            LOGe("llama_decode(step) failed");
            throw std::runtime_error("llama_decode() failed (step)");
        }
        _nCtxUsed += 1;
    } else {
        LOGi("[STEP] first token: using logits from prompt");
    }

    if (!_sampler || !_ctx) {
        throw std::runtime_error("Sampler/Context is null");
    }

    // 샘플링
    _currToken = llama_sampler_sample(_sampler, _ctx, /*pos=*/-1);

    // EOG 검사
    const llama_vocab* vocab = llama_model_get_vocab(_model);
    const int   vocab_size = llama_vocab_n_tokens(vocab);
    if (_currToken < 0 || _currToken >= vocab_size) {
        return "[EOG]";
    }
    if (llama_vocab_is_eog(vocab, _currToken)) {
        return "[EOG]";
    }

    // 토큰 → 텍스트 (C API: 버퍼로 받아 길이 리턴)
    char buf[512];
    int n = llama_token_to_piece(vocab, _currToken, buf, (int)sizeof(buf), /*special=*/true);
    std::string piece = (n > 0) ? std::string(buf, n) : std::string();

    // 누적
    _response += piece;
    _cacheResponseTokens.push_back(_currToken);
    _responseNumTokens += 1;

    const auto t1 = ggml_time_us();
    _responseGenerationTime += (t1 - t0);

    // 주기적 로깅(선택)
    if (_responseNumTokens % 10 == 0 || _nCtxUsed > (int)contextSize - 100) {
        LOGi("gen#=%ld  ctx_used=%d/%d(%.1f%%)",
             _responseNumTokens, _nCtxUsed, contextSize, _nCtxUsed * 100.0f / contextSize);
    }

    // 컨텍스트 초과 방지
    if (_nCtxUsed + 1 > (int)contextSize) {
        LOGe("Context size reached: %d/%d", _nCtxUsed + 1, contextSize);
        return "[EOG]";
    }

    return piece;
}

void
LLMInference::stopCompletion() {
    if (_storeChats) {
        addChatMessage(_response.c_str(), "assistant");
    }

    // 다음 턴을 위해 가벼운 상태만 리셋(캐시는 굳이 deep clear 안 함)
    _batch.n_tokens = 0;
    _response.clear();
    _cacheResponseTokens.clear();
    _responseNumTokens = 0;
    _responseGenerationTime = 0;

    if (_sampler) {
        llama_sampler_reset(_sampler);
    }

    // 템플릿 오프셋 갱신(assistant 미포함)
    const char* tmpl = llama_model_chat_template(_model, nullptr);
    _prevLen = llama_chat_apply_template(tmpl, _messages.data(), _messages.size(),
                                         /*add_assistant=*/false, nullptr, 0);
    if (_prevLen < 0) {
        throw std::runtime_error("llama_chat_apply_template() failed in stopCompletion()");
    }

    LOGi("Completion stopped (light reset).");
}

void
LLMInference::clearContext() {
    LOGi("Clearing context & memory (single-turn QA)");

    if (_ctx) {
        llama_kv_cache_clear(_ctx);
        llama_kv_cache_clear_drafts(_ctx);
    }

    _messages.clear();
    _formattedMessages.clear();
    _promptTokens.clear();
    _response.clear();
    _cacheResponseTokens.clear();

    _nCtxUsed = 0;
    _prevLen = 0;
    _responseNumTokens = 0;
    _responseGenerationTime = 0;

    _batch.n_tokens = 0;

    if (_sampler) {
        llama_sampler_reset(_sampler);
    }

    const char* tmpl = llama_model_chat_template(_model, nullptr);
    _prevLen = llama_chat_apply_template(tmpl, _messages.data(), _messages.size(),
                                         /*add_assistant=*/false, nullptr, 0);
    if (_prevLen < 0) {
        throw std::runtime_error("llama_chat_apply_template() failed in clearContext()");
    }
}
