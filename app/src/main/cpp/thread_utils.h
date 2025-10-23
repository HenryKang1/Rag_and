#ifndef THREAD_UTILS_H
#define THREAD_UTILS_H

#include <pthread.h>
#include <android/log.h>
#include <limits.h>

#define TAG "ThreadUtils"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static void increaseStackSize() {
    pthread_attr_t attr;
    size_t stacksize;
    
    // Get current stack size
    pthread_attr_init(&attr);
    pthread_attr_getstacksize(&attr, &stacksize);
    LOGI("Current stack size: %zu bytes (%.2f MB)", stacksize, stacksize / (1024.0 * 1024.0));
    
    // Set new stack size to 8MB
    size_t newStackSize = 8 * 1024 * 1024; // 8MB
    int result = pthread_attr_setstacksize(&attr, newStackSize);
    
    if (result == 0) {
        LOGI("Stack size increased to: %zu bytes (%.2f MB)", newStackSize, newStackSize / (1024.0 * 1024.0));
    } else {
        LOGE("Failed to set stack size: %d", result);
    }
    
    pthread_attr_destroy(&attr);
}

static void checkStackInfo() {
    pthread_attr_t attr;
    void *stackaddr;
    size_t stacksize;
    
    pthread_attr_init(&attr);
    
    // Get stack info for current thread
    if (pthread_getattr_np(pthread_self(), &attr) == 0) {
        pthread_attr_getstack(&attr, &stackaddr, &stacksize);
        LOGI("Thread stack info:");
        LOGI("  Stack address: %p", stackaddr);
        LOGI("  Stack size: %zu bytes (%.2f MB)", stacksize, stacksize / (1024.0 * 1024.0));
        
        // Check stack limit
        struct rlimit rlim;
        if (getrlimit(RLIMIT_STACK, &rlim) == 0) {
            LOGI("  Stack limit - soft: %llu, hard: %llu", 
                 (unsigned long long)rlim.rlim_cur, 
                 (unsigned long long)rlim.rlim_max);
        }
    }
    
    pthread_attr_destroy(&attr);
}

#endif // THREAD_UTILS_H