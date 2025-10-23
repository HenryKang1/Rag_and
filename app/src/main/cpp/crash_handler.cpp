#include <jni.h>
#include <android/log.h>
#include <signal.h>
#include <unwind.h>
#include <dlfcn.h>
#include <cxxabi.h>
#include <sstream>
#include <iomanip>

#define TAG "CrashHandler"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

struct BacktraceState {
    void** current;
    void** end;
};

static _Unwind_Reason_Code unwind_callback(struct _Unwind_Context* context, void* arg) {
    BacktraceState* state = static_cast<BacktraceState*>(arg);
    uintptr_t pc = _Unwind_GetIP(context);
    if (pc) {
        if (state->current == state->end) {
            return _URC_END_OF_STACK;
        } else {
            *state->current++ = reinterpret_cast<void*>(pc);
        }
    }
    return _URC_NO_REASON;
}

static size_t capture_backtrace(void** buffer, size_t max) {
    BacktraceState state = {buffer, buffer + max};
    _Unwind_Backtrace(unwind_callback, &state);
    return state.current - buffer;
}

static void dump_backtrace(int signum, const char* signal_name) {
    LOGE("==========================================");
    LOGE("Native crash detected!");
    LOGE("Signal: %d (%s)", signum, signal_name);
    LOGE("==========================================");
    
    const size_t max_frames = 32;
    void* buffer[max_frames];
    
    size_t frame_count = capture_backtrace(buffer, max_frames);
    
    LOGE("Backtrace (%zu frames):", frame_count);
    
    for (size_t idx = 0; idx < frame_count; ++idx) {
        const void* addr = buffer[idx];
        
        Dl_info info;
        if (dladdr(addr, &info) && info.dli_sname) {
            // Attempt to demangle C++ names
            int status = 0;
            char* demangled = abi::__cxa_demangle(info.dli_sname, nullptr, nullptr, &status);
            
            if (status == 0 && demangled) {
                LOGE("  #%02zu pc %p  %s (%s+%p)",
                     idx,
                     addr,
                     info.dli_fname,
                     demangled,
                     reinterpret_cast<void*>(
                         reinterpret_cast<uintptr_t>(addr) - 
                         reinterpret_cast<uintptr_t>(info.dli_saddr)
                     ));
                free(demangled);
            } else {
                LOGE("  #%02zu pc %p  %s (%s+%p)",
                     idx,
                     addr,
                     info.dli_fname,
                     info.dli_sname,
                     reinterpret_cast<void*>(
                         reinterpret_cast<uintptr_t>(addr) - 
                         reinterpret_cast<uintptr_t>(info.dli_saddr)
                     ));
            }
        } else {
            LOGE("  #%02zu pc %p", idx, addr);
        }
    }
    
    LOGE("==========================================");
}

static struct sigaction old_sa[NSIG];

static void signal_handler(int signum, siginfo_t* info, void* context) {
    const char* signal_name = "Unknown";
    
    switch(signum) {
        case SIGSEGV: signal_name = "SIGSEGV (Segmentation fault)"; break;
        case SIGABRT: signal_name = "SIGABRT (Abort)"; break;
        case SIGFPE:  signal_name = "SIGFPE (Floating point exception)"; break;
        case SIGILL:  signal_name = "SIGILL (Illegal instruction)"; break;
        case SIGBUS:  signal_name = "SIGBUS (Bus error)"; break;
        case SIGTRAP: signal_name = "SIGTRAP (Trap)"; break;
    }
    
    dump_backtrace(signum, signal_name);
    
    // Call the old handler
    if (old_sa[signum].sa_flags & SA_SIGINFO) {
        old_sa[signum].sa_sigaction(signum, info, context);
    } else if (old_sa[signum].sa_handler == SIG_DFL) {
        signal(signum, SIG_DFL);
        raise(signum);
    } else if (old_sa[signum].sa_handler != SIG_IGN) {
        old_sa[signum].sa_handler(signum);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_local_1llm_LlamaCppModel_nativeInstallCrashHandler(JNIEnv* env, jclass clazz) {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sigemptyset(&sa.sa_mask);
    sa.sa_sigaction = signal_handler;
    sa.sa_flags = SA_SIGINFO;
    
    int signals[] = {SIGSEGV, SIGABRT, SIGFPE, SIGILL, SIGBUS, SIGTRAP};
    
    for (int sig : signals) {
        if (sigaction(sig, &sa, &old_sa[sig]) == 0) {
            LOGE("Installed handler for signal %d", sig);
        }
    }
    
    LOGE("Native crash handler installed");
}