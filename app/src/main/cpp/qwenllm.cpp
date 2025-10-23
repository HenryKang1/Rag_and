#include <jni.h>
#include <android/log.h>
#include <string>
#include <memory>
#include <mutex>
#include <atomic>
#include <sys/resource.h>
#include "LLMInference.h"
#include "thread_utils.h"

#define TAG "QwenLLM-Native"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)

// Debug macro for detailed logging
#ifdef DEBUG
#define DEBUG_LOG(...) LOGD(__VA_ARGS__)
#else
#define DEBUG_LOG(...)
#endif

// Helper to check JNI exceptions
static bool checkJNIException(JNIEnv *env, const char* location) {
    if (env->ExceptionCheck()) {
        LOGE("JNI Exception at %s", location);
        env->ExceptionDescribe();
        env->ExceptionClear();
        return true;
    }
    return false;
}

// Thread safety wrapper for LLMInference
struct SafeLLMInference {
    std::unique_ptr<LLMInference> inference;
    std::mutex mutex;
    std::atomic<bool> isGenerating{false};
    
    SafeLLMInference() : inference(std::make_unique<LLMInference>()) {}
};

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_example_local_1llm_LlamaCppModel_nativeLoadModel(
    JNIEnv *env, jobject /* this */,
    jstring model_path,
    jint context_size,
    jint n_threads,
    jfloat temperature,
    jfloat min_p) {
    
    const char *path = env->GetStringUTFChars(model_path, nullptr);
    LOGI("nativeLoadModel called");
    
    // Check and log stack size
    checkStackInfo();
    
    // Try to increase stack size for better stability
    struct rlimit rl;
    getrlimit(RLIMIT_STACK, &rl);
    rl.rlim_cur = 8 * 1024 * 1024; // 8MB
    if (setrlimit(RLIMIT_STACK, &rl) == 0) {
        LOGI("Stack limit increased to 8MB");
    }
    
    LOGI("Loading model from: %s", path);
    LOGI("Parameters: context_size=%d, n_threads=%d, temp=%f, min_p=%f", 
         context_size, n_threads, temperature, min_p);
    
    // Check if file exists
    FILE *file = fopen(path, "r");
    if (file) {
        fseek(file, 0, SEEK_END);
        long size = ftell(file);
        fclose(file);
        LOGI("Model file exists, size: %ld bytes", size);
    } else {
        LOGE("Model file does not exist or cannot be opened: %s", path);
        env->ReleaseStringUTFChars(model_path, path);
        return 0;
    }
    
    try {
        LOGI("Creating SafeLLMInference instance");
        auto *safeInference = new SafeLLMInference();
        
        if (!safeInference || !safeInference->inference) {
            LOGE("Failed to create SafeLLMInference instance");
            env->ReleaseStringUTFChars(model_path, path);
            delete safeInference;
            return 0;
        }
        
        LOGI("SafeLLMInference instance created, calling loadModel");
        
        // Chat template for Qwen3 - pass empty string to use model's default template
        const char *chat_template = "";
        
        try {
            safeInference->inference->loadModel(
                path,
                min_p,
                temperature,
                false, // store_chats - set to false for single-turn QA
                context_size,
                chat_template,
                n_threads,
                true,  // use_mmap
                false, // use_mlock
                0.9f,  // top_p
                40,    // top_k
                0.0f,  // xtc_p
                1.0f   // xtc_t
            );
        } catch (const std::exception &e) {
            LOGE("loadModel threw exception: %s", e.what());
            delete safeInference;
            env->ReleaseStringUTFChars(model_path, path);
            return 0;
        }
        
        env->ReleaseStringUTFChars(model_path, path);
        LOGI("Model loaded successfully, returning pointer: %p", safeInference);
        return reinterpret_cast<jlong>(safeInference);
        
    } catch (const std::exception &e) {
        env->ReleaseStringUTFChars(model_path, path);
        LOGE("Failed to load model: %s", e.what());
        return 0;
    }
}

JNIEXPORT jboolean JNICALL
Java_com_example_local_1llm_LlamaCppModel_nativeStartGeneration(
    JNIEnv *env, jobject /* this */,
    jlong ptr,
    jstring prompt,
    jint max_tokens) {
    
    LOGI("nativeStartGeneration called with ptr=%p, max_tokens=%d", (void*)ptr, max_tokens);
    
    if (!ptr) {
        LOGE("nativeStartGeneration: Invalid model pointer");
        return JNI_FALSE;
    }
    
    if (!prompt) {
        LOGE("nativeStartGeneration: null prompt");
        return JNI_FALSE;
    }
    
    auto *safeInference = reinterpret_cast<SafeLLMInference *>(ptr);
    
    // Validate object state
    if (!safeInference->inference) {
        LOGE("nativeStartGeneration: inference object is null");
        return JNI_FALSE;
    }
    
    const char *prompt_str = env->GetStringUTFChars(prompt, nullptr);
    if (!prompt_str) {
        LOGE("nativeStartGeneration: Failed to get prompt string");
        return JNI_FALSE;
    }
    
    LOGI("Prompt length: %zu chars", strlen(prompt_str));
    DEBUG_LOG("Prompt preview: %.100s...", prompt_str);
    
    try {
        std::lock_guard<std::mutex> lock(safeInference->mutex);
        
        // If already generating, stop the current generation first
        if (safeInference->isGenerating) {
            LOGW("Generation already in progress, stopping current generation");
            try {
                safeInference->inference->stopCompletion();
                safeInference->isGenerating = false;
                LOGI("Previous generation stopped");
            } catch (const std::exception &e) {
                LOGW("Error stopping previous generation: %s", e.what());
            }
        }
        
        LOGI("Starting completion...");
        safeInference->inference->startCompletion(prompt_str);
        safeInference->isGenerating = true;
        env->ReleaseStringUTFChars(prompt, prompt_str);
        LOGI("Completion started successfully");
        return JNI_TRUE;
    } catch (const std::exception &e) {
        env->ReleaseStringUTFChars(prompt, prompt_str);
        LOGE("Failed to start generation: %s", e.what());
        
        // Throw Java exception
        jclass exClass = env->FindClass("java/lang/RuntimeException");
        if (exClass != nullptr) {
            env->ThrowNew(exClass, e.what());
        }
        return JNI_FALSE;
    } catch (...) {
        env->ReleaseStringUTFChars(prompt, prompt_str);
        LOGE("Unknown exception in startGeneration");
        return JNI_FALSE;
    }
}

JNIEXPORT jstring JNICALL
Java_com_example_local_1llm_LlamaCppModel_nativeGetNextToken(
    JNIEnv *env, jobject /* this */,
    jlong ptr) {
    
    LOGI("nativeGetNextToken called with ptr=%p", (void*)ptr);
    
    if (!ptr) {
        LOGW("nativeGetNextToken: null pointer provided");
        return env->NewStringUTF("");
    }
    
    // Validate pointer alignment
    if (ptr % sizeof(void*) != 0) {
        LOGE("nativeGetNextToken: misaligned pointer %p", (void*)ptr);
        return env->NewStringUTF("");
    }
    
    auto *safeInference = reinterpret_cast<SafeLLMInference *>(ptr);
    
    // Validate object state
    if (!safeInference->inference) {
        LOGE("nativeGetNextToken: inference object is null");
        return env->NewStringUTF("");
    }
    
    try {
        std::lock_guard<std::mutex> lock(safeInference->mutex);
        
        // Check if generation is active
        if (!safeInference->isGenerating) {
            DEBUG_LOG("nativeGetNextToken: generation not active");
            return env->NewStringUTF("");
        }
        
        LOGI("Calling completionLoop...");
        std::string token = safeInference->inference->completionLoop();
        LOGI("completionLoop returned: '%s' (length=%zu)", token.c_str(), token.length());
        
        // Check for end of generation
        if (token == "[EOG]" || token.empty()) {
            LOGI("Generation completed, cleaning up context");
            try {
                safeInference->inference->stopCompletion();
            } catch (const std::exception &e) {
                LOGW("Error stopping completion: %s", e.what());
            }
            safeInference->isGenerating = false;
            return env->NewStringUTF("");
        }
        
        // Create Java string and check for exceptions
        jstring result = env->NewStringUTF(token.c_str());
        if (checkJNIException(env, "NewStringUTF")) {
            safeInference->isGenerating = false;
            return env->NewStringUTF("");
        }
        
        return result;
    } catch (const std::runtime_error &e) {
        // This likely means the generation was interrupted/stopped
        LOGW("Runtime error in nativeGetNextToken (likely generation stopped): %s", e.what());
        safeInference->isGenerating = false;
        // Don't throw exception, just return empty string
        return env->NewStringUTF("");
    } catch (const std::exception &e) {
        LOGW("Exception in nativeGetNextToken (likely generation stopped): %s", e.what());
        safeInference->isGenerating = false;
        return env->NewStringUTF("");
    } catch (...) {
        LOGW("Unknown exception in nativeGetNextToken");
        safeInference->isGenerating = false;
        return env->NewStringUTF("");
    }
}

JNIEXPORT jstring JNICALL
Java_com_example_local_1llm_LlamaCppModel_nativeGenerateComplete(
    JNIEnv *env, jobject /* this */,
    jlong ptr,
    jstring prompt,
    jint max_tokens) {
    
    if (!ptr) {
        LOGE("Invalid model pointer");
        return env->NewStringUTF("");
    }
    
    auto *safeInference = reinterpret_cast<SafeLLMInference *>(ptr);
    const char *prompt_str = env->GetStringUTFChars(prompt, nullptr);
    
    try {
        std::lock_guard<std::mutex> lock(safeInference->mutex);
        
        // If already generating, stop the current generation first
        if (safeInference->isGenerating) {
            LOGW("Generation already in progress, stopping current generation");
            try {
                safeInference->inference->stopCompletion();
                safeInference->isGenerating = false;
                LOGI("Previous generation stopped");
            } catch (const std::exception &e) {
                LOGW("Error stopping previous generation: %s", e.what());
            }
        }
        
        // Start generation
        safeInference->inference->startCompletion(prompt_str);
        safeInference->isGenerating = true;
        env->ReleaseStringUTFChars(prompt, prompt_str);
        
        // Collect all tokens
        std::string response;
        int token_count = 0;
        
        while (token_count < max_tokens) {
            std::string token = safeInference->inference->completionLoop();
            if (token.empty() || token == "[EOG]") {
                break;
            }
            response += token;
            token_count++;
        }
        
        safeInference->isGenerating = false;
        LOGI("Generated %d tokens", token_count);
        return env->NewStringUTF(response.c_str());
        
    } catch (const std::exception &e) {
        safeInference->isGenerating = false;
        env->ReleaseStringUTFChars(prompt, prompt_str);
        LOGE("Failed to generate response: %s", e.what());
        return env->NewStringUTF("");
    }
}

JNIEXPORT void JNICALL
Java_com_example_local_1llm_LlamaCppModel_nativeReset(
    JNIEnv *env, jobject /* this */,
    jlong ptr) {
    
    LOGI("nativeReset called with ptr=%p", (void*)ptr);
    
    if (!ptr) {
        LOGE("nativeReset: Invalid model pointer");
        return;
    }
    
    auto *safeInference = reinterpret_cast<SafeLLMInference *>(ptr);
    
    try {
        std::lock_guard<std::mutex> lock(safeInference->mutex);
        
        // Stop any ongoing generation
        if (safeInference->isGenerating) {
            safeInference->inference->stopCompletion();
            safeInference->isGenerating = false;
        }
        
        // Clear the context and free memory using the public method
        if (safeInference->inference) {
            LOGI("Calling clearContext to free memory");
            safeInference->inference->clearContext();
        }
        
        LOGI("Model state reset successfully");
    } catch (const std::exception &e) {
        LOGE("Failed to reset model state: %s", e.what());
    }
}

JNIEXPORT void JNICALL
Java_com_example_local_1llm_LlamaCppModel_nativeRelease(
    JNIEnv *env, jobject /* this */,
    jlong ptr) {
    
    if (ptr) {
        auto *safeInference = reinterpret_cast<SafeLLMInference *>(ptr);
        delete safeInference;
        LOGI("Model released");
    }
}

} // extern "C"